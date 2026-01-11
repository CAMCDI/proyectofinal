#!/bin/sh
# Based on Deno installer: Copyright 2019 the Deno authors. All rights reserved. MIT license.

set -e

main() {
	os=$(uname -s)
	arch=$(uname -m)
	version=${1:-latest}
	
	github_uri="https://github.com/tunneltodev/tunnelto/releases"
	
	uri="https://github.com/tunneltodev/tunnelto/releases/${version}/download/$os-$arch.tar.gz"
	uri_check=$(curl -o /dev/null -sIL -w "%{http_code}" https://github.com/tunneltodev/tunnelto/releases/${version}/download/$os-$arch.tar.gz)
	if [ "$uri_check" != "200" ]; then
		echo "Error: Unable to find a release for $os-$arch - see $github_uri for all versions" 1>&2
		exit 1
	else
		echo "Found release for $os-$arch:$version"
	fi
    binary_name="tunnelto"
    sim_binary_name="tunn"

	install_dir="${TUNNELTO_INSTALL:-$HOME/.tunnelto}"

	bin_dir="$install_dir/bin"
	tmp_dir="$install_dir/tmp"

	exe="$bin_dir/$binary_name"
	simexe="$bin_dir/$sim_binary_name"

	mkdir -p "$bin_dir"
	mkdir -p "$tmp_dir"

	curl -q --fail --location --progress-bar --output "$tmp_dir/$binary_name.tar.gz" "$uri"

	# extract to tmp dir so we don't open existing executable file for writing:
	tar -C "$tmp_dir" -xzf "$tmp_dir/$binary_name.tar.gz"
	bin_path="$tmp_dir/app/$binary_name"
	
	chmod +x "$bin_path"
	# atomically rename into place:
	mv "$bin_path" "$exe"

	rm "$tmp_dir/$binary_name.tar.gz"
	rm -rf $bin_path

	ln -sf $exe $simexe

	"$exe" --version
	
	echo "\n$binary_name was installed successfully to $exe\n"

	case $SHELL in
	/bin/zsh) shell_profile=".zshrc" ;;
	*) shell_profile=".bash_profile" ;;
	esac
	echo "Manually add the directory to your \$HOME/$shell_profile (or similar)"
	echo "  export TUNNELTO_INSTALL=\"$install_dir\""
	echo "  export PATH=\"\$TUNNELTO_INSTALL/bin:\$PATH\""
	echo "\nRun '$exe --help' to get started"

	if command -v $binary_name >/dev/null; then
		current_version=$(which "$binary_name")		
		if [ "$current_version" != "$exe" ]; then
			echo "\nWARNING: there is an existing installation of '$binary_name' at "$current_version" that conflicts with this install ($exe).\n"
		fi
	fi

}

main "$1"