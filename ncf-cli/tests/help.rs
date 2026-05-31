use std::process::Command;

#[test]
fn cli_prints_help() {
    // Cargo sets CARGO_BIN_EXE_<name> env for integration tests; use it to run the built binary
    let exe = std::env::var("CARGO_BIN_EXE_ncf-cli").expect("CARGO_BIN_EXE_ncf-cli not set");
    let output = Command::new(exe).arg("--help").output().expect("failed to spawn");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("NCF CLI tool") || stdout.contains("USAGE") || stdout.len() > 0);
}
