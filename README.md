# Deploy PyGame Games to a Website

Run your PyGame games on the web with Github pages. You game will actually be
running entirely in the browser using WebAssembly and pygbag. 

## Setup Environment for local development

First, copy this repo. You should use the green 
<span style="background-color: green; color: white; padding: 10px; font-size: 16px;">"Use this template ▾"</span> 
 button, but you can also fork the repository to your own account. 

Clone to your development environment,  or make a codespace, then reate an
environment and load the requirements.

```bash
    python3 -mvenv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
```

Then you can run the game:

```bash
 python src/main.py
```

After demonstrating that the Space Invaders game works, you can develop your own
game in the `src/` directory. 

## Run the web game

```bash
pygbag src/
```
Then you can open [http://localhost:8000/](http://localhost:8000/) to play the game

## Publish to Github

You should have this repo in a Github repo. Go to `Settings>Pages`. Under "Build
and deployment" set "Source" to "Deploy from Branch"

Then publish the branch with: `make publish`

Wait a bit then back in the Github Pages settings, set "Branch" to "gh-pages" (
this branch is created by `make publish` and may take a while to appear ) Wait a
bit more, and when your pages are readh, you will see the URL at the top of the
Github Settings page. 


## Example

You can play the [demo Space Invaders game  here](https://league-curriculum.github.io/Python-Web-Game/). 


## Original 

This repo is copied from the original by Santhoshkumard11, from this [Github repo](https://github.com/Santhoshkumard11/deploy-pygame). See his [DEV article for a discussion](https://dev.to/sandy_codes_py/deploy-pygames-to-github-pages-with-webassembly-56po)

-------

Development of The LEAGUE’s curriculum is generously funded by the Itzkowitz Family Foundation.
