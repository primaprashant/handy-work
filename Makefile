.PHONY: personal-setup personal-build personal-install personal-status personal-reset-permissions

# One-time setup: install dependencies and create/check a stable local signing identity.
personal-setup:
	@./scripts/macos-personal.sh setup

# Build a release Handy.app signed with the stable personal identity.
personal-build:
	@./scripts/macos-personal.sh build

# Normal repeat command: build, replace /Applications/Handy.app, and launch it.
personal-install:
	@./scripts/macos-personal.sh install

# Show the fork revision and the signing identities of built/installed bundles.
personal-status:
	@./scripts/macos-personal.sh status

# Recovery command for a stale macOS TCC grant. Normal installs detect this automatically.
personal-reset-permissions:
	@./scripts/macos-personal.sh reset-permissions
