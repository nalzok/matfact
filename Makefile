.PHONY: run

run:
	pipenv run python3 -m lnn.cli --ground_truth reflector --p 8 --features 8 --features 8 --features 8 --epochs 8192
