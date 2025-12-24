$ErrorActionPreference = "Stop"

npm run build

git add -A
git commit -m "Deploy site"
git push

npx gh-pages -d build
