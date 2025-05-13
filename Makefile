.PHONY: wheel docker deploy clean upload

VERSION := $(shell python setup.py --version)
DOCKER_TAG := $(shell echo $(VERSION) | tr '+.' '__-')

IMAGE := jsg-ml:$(DOCKER_TAG)

wheel:
	@echo "버전 $(VERSION)의 wheel 파일을 생성합니다..."
	python setup.py bdist_wheel

docker: clean wheel
	@echo "도커 이미지를 생성합니다. 태그: $(IMAGE)"
	docker build -t $(IMAGE) .

deploy: docker
	@echo "포트 8000을 사용하는 컨테이너가 있는지 확인합니다..."
	@PORT_CONTAINER=$$(docker ps --filter "publish=8000" --format "{{.ID}}"); \
	if [ ! -z "$$PORT_CONTAINER" ]; then \
		echo "포트 8000을 사용하는 컨테이너 $$PORT_CONTAINER 를 중지하고 삭제합니다..."; \
		docker rm -f $$PORT_CONTAINER; \
	else \
		echo "포트 8000을 사용하는 컨테이너가 없습니다."; \
	fi
	@echo "'jsg-ml-running' 이름의 컨테이너가 존재하는지 확인합니다..."
	@NAME_CONTAINER=$$(docker ps -aqf "name=jsg-ml-running"); \
	if [ ! -z "$$NAME_CONTAINER" ]; then \
		echo "'jsg-ml-running' 컨테이너를 삭제합니다..."; \
		docker rm -f $$NAME_CONTAINER; \
	else \
		echo "'jsg-ml-running' 컨테이너가 존재하지 않습니다."; \
	fi
	@echo "도커 컨테이너를 실행합니다. 이미지: $(IMAGE)"
	docker run -d -p 8000:8000 --name jsg-ml-running $(IMAGE)

clean:
	@echo "빌드 결과물을 정리합니다..."
	rm -rf build dist *.egg-info
