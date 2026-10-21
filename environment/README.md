# Environment capture

Run `capture_environment.sh` inside the exact container used for final analyses. Archive the generated inventories before release.

The immutable Docker image digest must be obtained from the host or registry; it generally cannot be inferred reliably from inside the running container.
