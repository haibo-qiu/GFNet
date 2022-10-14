# How to Create a Project Page [![Project](https://img.shields.io/badge/Project-Page-important)](https://haibo-qiu.github.io/GFNet/)

The key operation is creating a gh-pages branch in the target repo, then the github-pages action will automatically deploy it.

1. Checkout an orphan gh-pages branch
    ```bash
    git checkout --orphan gh-pages
    git reset --hard
    ```

2. Add the project content (with the template like [Nerfies website](https://nerfies.github.io))

3. (optional) Locally test with [jekyll and bundler](https://jekyllrb.com/tutorials/using-jekyll-with-bundler/) 
    ```bash
    # simple cmd
    bundle init
    bundle config set --local path 'vendor/bundle'
    bundle add jekyll
    bundle install
    bundle exec jekyll serve
    ```

4. Commit and push
    ```bash
    git commit -am "init commit"
    git push origin gh-pages
    ```
5. Access to the project page in `https://xxx.github.io/YOUR_REPO_NAME/`
