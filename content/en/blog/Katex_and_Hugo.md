---
author: "Satyan Sharma"
title: "Inline Math with KaTeX and Hugo"
date: 2021-07-13
math: true
tags: ["Katex", "hugo"]
---

While writing math in this blog I realized that inline math does not render. 
After some search on [Partials](https://gohugo.io/templates/partials/) and 
[Katex](https://katex.org/docs/options.html), I found out this solution to work for me. 

Now I can write `$\int \oint \sum \prod$` to render $\int \oint \sum \prod$ and `$$\int \oint \sum \prod$$` to render as equation
$$\int \oint \sum \prod$$

This is the partial that I am using to set up the delimiters 
option so as to use $$ for math in display mode and $ for inline math.

```html
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ]
        });
    });
</script>
```

