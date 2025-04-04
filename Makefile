.PHONY: wheel docker deploy clean

wheel:
	@echo "Building wheel file..."
	python setup.py bdist_wheel

docker: wheel
	@echo "Building Docker image..."
	docker build -t jsg-ml:latest .

deploy: docker
	@echo "Deploying Docker container..."
	docker run --rm -d -p 8000:8000 jsg-ml:latest

clean:
	@echo "Cleaning up..."
	rm -rf build dist *.egg-info
