#!/usr/bin/env python3
"""Enable validated decode flags without replacing the machine's launch profile."""
import argparse
import re
import subprocess
from pathlib import Path

FLAGS = {'FNX_INDEX_POST': '1'}

def patch_profile(data, flags=None):
    flags = FLAGS if flags is None else flags
    text = data.decode('utf-8')
    newline = '\r\n' if '\r\n' in text else '\n'
    text = text.replace('\r\n', '\n')
    anchor = re.compile(r'^FNX_CONV_PREFILL=\$\{FNX_CONV_PREFILL:-[01]\}[ \t]*$', re.M)
    if len(anchor.findall(text)) == 1:
        for name, value in flags.items():
            assert name in FLAGS and value in ('0', '1')
            declaration = re.compile(r'^' + name + r'=\$\{' + name + r':-[01]\}[ \t]*$', re.M)
            declarations = declaration.findall(text)
            assert len(declarations) <= 1, name + ': duplicate defaults'
            default = name + '=${' + name + ':-' + value + '}'
            if declarations:
                text = declaration.sub(lambda _: default, text)
            else:
                text = anchor.sub(lambda m: default + '\n' + m[0], text, count=1)
            forward = re.compile(r'\b' + name + r'=(?:\$\{' + name + r'\}|\$' + name + r')(?![A-Za-z0-9_])')
            literal = re.compile(r'\b' + name + r'=[01](?![A-Za-z0-9_])')
            existing = len(forward.findall(text)) + len(literal.findall(text))
            assert existing <= 1, name + ': duplicate command assignments'
            if literal.search(text):
                text = literal.sub(lambda _: name + '=$' + name, text)
            elif not forward.search(text):
                command_anchor = 'FNX_CONV_PREFILL=$FNX_CONV_PREFILL'
                assert text.count(command_anchor) == 1, 'unsupported parameterized command'
                text = text.replace(command_anchor, command_anchor + ' ' + name + '=$' + name)
            assert len(forward.findall(text)) == 1
    else:
        legacy = 'FNX_CONV_PREFILL=1'
        assert text.count(legacy) == 1, 'unsupported launch profile; file not changed'
        for name, value in flags.items():
            assert name in FLAGS and value in ('0', '1')
            literal = re.compile(r'\b' + name + r'=[01](?![A-Za-z0-9_])')
            assert len(literal.findall(text)) <= 1
            if literal.search(text):
                text = literal.sub(lambda _: name + '=' + value, text)
            else:
                assert name + '=' not in text, 'unsupported flag expression'
                text = text.replace(legacy, legacy + ' ' + name + '=' + value)
    return text.replace('\n', newline).encode('utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    original = args.profile.read_bytes()
    patched = patch_profile(original)
    subprocess.run(['bash', '-n'], input=patched, check=True)
    if not args.check:
        args.profile.write_bytes(patched)
    print('PROFILE_CHECK_OK' if args.check else 'PROFILE_UPDATED', args.profile)
