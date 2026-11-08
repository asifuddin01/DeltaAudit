# One-command contract (BUILD_GUIDE Part G). Targets beyond env/test/lint/info
# call CLI commands that land in later phases; they fail with "No such command" until then.
RUN := uv run

.PHONY: env info test lint data verify baselines train evaluate stats report all tiny colab-bundle

env:
	uv sync --frozen

info:
	$(RUN) deltaaudit info

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check . && $(RUN) black --check .

data:
	$(RUN) deltaaudit download --all
	$(RUN) deltaaudit verify
	$(RUN) deltaaudit card --all
	$(RUN) deltaaudit preprocess --all
	$(RUN) deltaaudit describe --all
	$(RUN) deltaaudit split --all
	$(RUN) deltaaudit verify

verify:
	$(RUN) deltaaudit verify

baselines: verify
	$(RUN) deltaaudit baselines

train: verify
	$(RUN) deltaaudit train

evaluate:
	$(RUN) deltaaudit evaluate

stats:
	$(RUN) deltaaudit stats

report:
	$(RUN) deltaaudit figures --all

all:
	$(RUN) deltaaudit all

tiny:
	$(RUN) deltaaudit all --tiny

colab-bundle:
	$(RUN) deltaaudit colab-bundle
