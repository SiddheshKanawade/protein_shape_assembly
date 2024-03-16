format:
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place ./ --exclude=__init__.py
	black ./ --line-length 80
	isort --profile black ./ --line-length 80