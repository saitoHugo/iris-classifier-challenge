dev:
	(cd iris-api/ && uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8080)
dev-docker:
	docker run -p 8080:8080 --name iris-api-container iris-api
docker-clean:
	# #TODO: need updte
	(docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) )
docker-build:
	docker build -t iris-api .

docker-run:
	docker run -p 8080:8080 --name iris-api-container iris-api

test:
	pytest -ra


