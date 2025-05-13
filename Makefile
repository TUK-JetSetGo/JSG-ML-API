.PHONY: wheel docker deploy clean upload

VERSION := $(shell python setup.py --version)
DOCKER_TAG := $(shell echo $(VERSION) | tr '+.' '__-')

IMAGE := jsg-ml:$(DOCKER_TAG)

wheel:
	@echo "🔧 Building wheel file (version: $(VERSION))..."
	python setup.py bdist_wheel

docker: clean wheel
	@echo "🐳 Building Docker image with tag $(IMAGE)..."
	docker build -t $(IMAGE) .

deploy: docker
	@echo "🚀 Deploying Docker container with image $(IMAGE)..."
	docker run --rm -d -p 8000:8000 $(IMAGE)

clean:
	@echo "🧹 Cleaning up build artifacts..."
	rm -rf build dist *.egg-info
