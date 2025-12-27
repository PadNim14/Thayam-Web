
# Infer the repository URL from the git remote
REPO_URL := $(shell git config --get remote.origin.url)

APP_NAME = 'Thayam'
ICON =

CNAME = 


publish:
	@pygbag --build --app_name $(APP_NAME) $(if $(ICON),--icon $(ICON),) src
	@echo "Publishing to $(REPO_URL)"
ifdef CNAME
	@ghp-import -n -p --cname $(CNAME) -f ./src/build/web 
else
	@ghp-import -n -p -f ./src/build/web 
endif

runweb:
	@pygbag src
