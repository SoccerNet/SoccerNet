#!/usr/bin/env python3
"""HEAD-probe one EXRCS ownCloud share token without downloading anything.

Read-only: sends a single HTTP HEAD request. Never mutates or deletes
anything on EXRCS.

Usage:
    python probe_share.py --token DxdZ39FT9GqCkEe --file train.zip --password SoccerNet
    python probe_share.py --token D4xoTmkC7SEJGZp --file test_secret.zip --password-env SOCCERNET_NDA_PASSWORD
"""
import argparse
import os
import urllib.error
import urllib.request

OWNCLOUD_BASE = "https://exrcsdrive.kaust.edu.sa/public.php/webdav/"


def probe(token, fname, password):
    url = OWNCLOUD_BASE + fname
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, OWNCLOUD_BASE, token, password)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", required=True, help="ownCloud share token (the 'user' in the old Downloader.py)")
    parser.add_argument("--file", required=True, help="filename to test, e.g. train.zip")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--password", help="literal password (fine for the public generic password only)")
    group.add_argument("--password-env", help="name of an env var holding the real NDA/reviewers password")
    args = parser.parse_args()

    password = args.password or os.environ[args.password_env]
    status = probe(args.token, args.file, password)
    print(f"{args.token}  {args.file}  -> {status}")


if __name__ == "__main__":
    main()
