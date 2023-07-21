# Causal ML: Drafting in Bookdown

This portion of the repo contains what we need to create our manuscript using RMarkdown and bookdown. We will draft our chapters in incrementally indexed RMarkdown (.Rmd) files, and our output (including HTML, PDF, and EPUB all named `main.*`) will be produced in the `_book` subdirectory.

## Installation and Setup

1. Install R and RStudio
2. Open RStudio and run `install.packages("bookdown")`
3. Install a LaTeX distribution: I used Yihue's TinyTex distribution
  * Run `install.packages("tinytex")`
  * Run `tinytex::install_tinytex()`
4. Restart RStudio and activate the project in this directory (`bookdown/`)
5. Test that this book renders
  * See the steps in `index.Rmd` for a description of how to render the book locally

## Additional resources

The **bookdown** book: <https://bookdown.org/yihui/bookdown/>

The **bookdown** GitHub repo: <https://github.com/rstudio/bookdown>

The **bookdown** package reference site: <https://pkgs.rstudio.com/bookdown>
