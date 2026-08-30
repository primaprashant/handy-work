#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SIGNING_IDENTITY="${HANDY_PERSONAL_SIGNING_IDENTITY:-Handy Personal Build}"
readonly BUNDLE_ID="com.pais.handy"
readonly BUILT_APP="${REPO_ROOT}/src-tauri/target/release/bundle/macos/Handy.app"
readonly INSTALLED_APP="/Applications/Handy.app"
readonly PERSONAL_CONFIG="src-tauri/tauri.personal.conf.json"
readonly FORK_REV="6300061f06ae7e918ac87c1f4907368effa829d6"

die() {
  echo "error: $*" >&2
  exit 1
}

note() {
  echo "==> $*"
}

require_macos() {
  [ "$(uname -s)" = "Darwin" ] || die "personal builds are supported only on macOS"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

identity_hash() {
  security find-identity -v -p codesigning 2>/dev/null |
    awk -v name="${SIGNING_IDENTITY}" 'index($0, "\"" name "\"") { print $2; exit }'
}

require_identity() {
  local hash
  hash="$(identity_hash)"
  if [ -z "${hash}" ]; then
    cat >&2 <<EOF
No valid code-signing identity named "${SIGNING_IDENTITY}" was found.

Run 'make personal-setup' and follow its one-time Keychain Access steps.
EOF
    exit 1
  fi
  printf '%s\n' "${hash}"
}

verify_fork_lock() {
  local lockfile="${REPO_ROOT}/src-tauri/Cargo.lock"
  local expected="git+https://github.com/primaprashant/transcribe.cpp.git?rev=${FORK_REV}#${FORK_REV}"
  grep -Fq "${expected}" "${lockfile}" || die \
    "Cargo.lock is not pinned to the expected transcribe.cpp fork revision"
}

bundle_identifier() {
  /usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$1/Contents/Info.plist" 2>/dev/null || true
}

designated_requirement() {
  codesign --display --requirements - "$1" 2>&1 |
    sed -n 's/^#* *designated => //p' |
    tail -n 1
}

signature_summary() {
  local app="$1"
  if [ ! -d "${app}" ]; then
    echo "missing"
    return
  fi

  local requirement
  requirement="$(designated_requirement "${app}")"
  if [ -z "${requirement}" ]; then
    echo "unsigned or invalid"
  else
    echo "${requirement}"
  fi
}

verify_app() {
  local app="$1"
  [ -d "${app}" ] || die "app bundle not found: ${app}"
  [ "$(bundle_identifier "${app}")" = "${BUNDLE_ID}" ] ||
    die "unexpected bundle identifier in ${app}"
  codesign --verify --deep --strict --verbose=2 "${app}"

  local details
  details="$(codesign --display --verbose=4 "${app}" 2>&1)"
  if printf '%s\n' "${details}" | grep -q '^Signature=adhoc$'; then
    die "${app} is ad-hoc signed; the stable personal identity was not applied"
  fi
}

setup_identity() {
  if [ -n "$(identity_hash)" ]; then
    note "Found signing identity: ${SIGNING_IDENTITY}"
    return
  fi

  cat <<EOF
Handy needs one stable signing identity so macOS recognizes rebuilt versions as
the same app. Keychain Access will open now. Create this development-only
self-signed certificate:

  Name:             ${SIGNING_IDENTITY}
  Identity Type:    Self Signed Root
  Certificate Type: Code Signing

In Keychain Access choose:
  Keychain Access > Certificate Assistant > Create a Certificate...

Enable "Let me override defaults", choose a long validity such as 3650 days,
accept the remaining defaults, and store it in the login keychain. This is only
for your own Mac; it is not suitable for distributing the app to other users.
EOF

  open -a "Keychain Access"
  if [ -t 0 ]; then
    printf '\nPress Return after the certificate has been created... '
    read -r _
  else
    die "create the certificate in Keychain Access, then rerun personal-setup"
  fi

  [ -n "$(identity_hash)" ] || die \
    "the identity is still unavailable; ensure it is in login > My Certificates and is trusted for code signing"
  note "Created signing identity: ${SIGNING_IDENTITY}"
}

setup() {
  require_macos
  require_command bun
  require_command cargo
  require_command cmake
  require_command codesign
  require_command security
  xcode-select -p >/dev/null 2>&1 || die "install Xcode Command Line Tools with: xcode-select --install"

  setup_identity
  note "Installing locked frontend dependencies"
  (cd "${REPO_ROOT}" && bun install --frozen-lockfile)

  note "One-time setup is complete"
  echo "Run 'make personal-install' whenever you want to rebuild and install Handy."
}

build_app() {
  require_macos
  require_command bun
  require_command cargo
  require_command cmake
  require_command codesign

  local hash
  hash="$(require_identity)"
  verify_fork_lock
  note "Building Handy with transcribe.cpp fork ${FORK_REV}"
  (
    cd "${REPO_ROOT}"
    APPLE_SIGNING_IDENTITY="${hash}" \
      bun run tauri build --bundles app --config "${PERSONAL_CONFIG}"
  )

  verify_app "${BUILT_APP}"
  note "Built and verified ${BUILT_APP}"
}

quit_installed_app() {
  osascript -e 'tell application id "com.pais.handy" to quit' >/dev/null 2>&1 || true

  local attempts=0
  while pgrep -f '^/Applications/Handy.app/Contents/MacOS/handy($| )' >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [ "${attempts}" -ge 20 ]; then
      die "Handy is still running; quit it and rerun personal-install"
    fi
    sleep 0.25
  done
}

replace_installed_app() {
  # Keep the destructive target literal and validate it before removing an old
  # bundle. Merging app directories can leave stale signed files behind.
  [ "${INSTALLED_APP}" = "/Applications/Handy.app" ] || die "refusing unexpected install path"

  if [ -w "/Applications" ]; then
    rm -rf -- "/Applications/Handy.app"
    ditto "${BUILT_APP}" "/Applications/Handy.app"
  else
    note "Administrator permission is required to write to /Applications"
    sudo rm -rf -- "/Applications/Handy.app"
    sudo ditto "${BUILT_APP}" "/Applications/Handy.app"
  fi
}

reset_permissions() {
  require_macos
  note "Resetting only Handy's Accessibility and Microphone grants"
  tccutil reset Accessibility "${BUNDLE_ID}"
  tccutil reset Microphone "${BUNDLE_ID}"
  echo "Open Handy and grant the two permissions once when macOS asks."
}

install_app() {
  local old_exists=false
  local old_requirement=""
  if [ -d "${INSTALLED_APP}" ]; then
    old_exists=true
    old_requirement="$(designated_requirement "${INSTALLED_APP}")"
  fi

  build_app

  local new_requirement
  new_requirement="$(designated_requirement "${BUILT_APP}")"
  [ -n "${new_requirement}" ] || die "could not read the new app's designated requirement"

  quit_installed_app
  replace_installed_app
  verify_app "${INSTALLED_APP}"

  if [ "${old_exists}" = true ] && [ "${old_requirement}" != "${new_requirement}" ]; then
    note "The app signing identity changed; macOS permissions must be granted once for the personal build"
    reset_permissions
  else
    note "The signing identity is unchanged; preserving existing macOS permissions"
  fi

  note "Installed ${INSTALLED_APP}"
  open "${INSTALLED_APP}"
}

status() {
  require_macos
  echo "Fork revision:      ${FORK_REV}"
  echo "Signing identity:   ${SIGNING_IDENTITY}"
  if [ -n "$(identity_hash)" ]; then
    echo "Identity SHA-1:      $(identity_hash)"
  else
    echo "Identity SHA-1:      missing"
  fi
  echo "Built requirement:  $(signature_summary "${BUILT_APP}")"
  echo "Installed req.:     $(signature_summary "${INSTALLED_APP}")"
}

usage() {
  cat <<EOF
Usage: $0 setup|build|install|status|reset-permissions

  setup              one-time dependency and signing-identity setup
  build              build a signed release app without installing it
  install            build, install to /Applications, and launch
  status             show fork and code-signing status
  reset-permissions  manually reset Handy's Accessibility and Microphone grants
EOF
}

case "${1:-}" in
  setup) setup ;;
  build) build_app ;;
  install) install_app ;;
  status) status ;;
  reset-permissions) reset_permissions ;;
  *) usage; exit 2 ;;
esac
