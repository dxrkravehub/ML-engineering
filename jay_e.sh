for f in $(git ls-files | grep ":"); do
  new=$(echo "$f" | tr ':' '_')
  echo "$f -> $new"
done
