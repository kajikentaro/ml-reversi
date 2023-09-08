backend-dev:
	uvicorn api:app --reload

exec:
	docker compose run dev bash