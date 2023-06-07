#!/usr/bin/env bash

set -e

rm -rf dist

python -m build
echo "Push to pypi with: python -m twine upload --repository pypi dist/*"