.PHONY: docker deploy clean

VERSION := $(shell git rev-parse --short HEAD)
IMAGE := jsg-ml:$(VERSION)

docker:
	@echo "도커 이미지를 빌드합니다..."
	docker build -t $(IMAGE) .

deploy: docker
	@echo "포트 8000을 사용하는 컨테이너가 있는지 확인합니다..."
	@PORT_CONTAINER=$$(docker ps --filter "publish=8000" --format "{{.ID}}"); \
	if [ ! -z "$$PORT_CONTAINER" ]; then \
		echo "기존 컨테이너 $$PORT_CONTAINER 중지 및 제거..."; \
		docker rm -f $$PORT_CONTAINER; \
	fi
	@NAME_CONTAINER=$$(docker ps -aqf "name=jsg-ml-running"); \
	if [ ! -z "$$NAME_CONTAINER" ]; then \
		echo "이름 jsg-ml-running 컨테이너 제거..."; \
		docker rm -f $$NAME_CONTAINER; \
	fi
	@echo "도커 컨테이너 실행..."
	docker run -d -p 8000:8000 --name jsg-ml-running $(IMAGE)

clean:
	@echo "빌드 캐시 및 불필요 파일 정리..."
	rm -rf __pycache__ .pytest_cache
