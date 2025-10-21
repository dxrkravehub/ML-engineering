#!/bin/bash
for f in $(git ls-files | grep ":"); do
  new=$(echo "$f" | tr ':' '_')
  echo "Renaming $f -> $new"
  git mv "$f" "$new"
done

git commit -m "rename all ':' to '_'"
git push
