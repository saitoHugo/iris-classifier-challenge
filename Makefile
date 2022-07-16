dev:
	(cd iris-api/ && uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8080)

docker-build:
	docker build -t iris-api .

docker-run:
	docker run -p 8080:8080 --name iris-api-container iris-api

docker-clean:
	(docker stop iris-api-container && docker rm iris-api-container )

heroku-docker-build:
	docker build -t registry.heroku.com/iris-classifier-challenge/web .

heroku-docker-run:
	docker run -d -p 8080:8080 --name iris-api-container  registry.heroku.com/iris-classifier-challenge/web:latest
	
heroku-docker-registry:
	docker push registry.heroku.com/iris-classifier-challenge/web

heroku-release:
	heroku container:release -a iris-classifier-challenge web

test:
	pytest -ra


