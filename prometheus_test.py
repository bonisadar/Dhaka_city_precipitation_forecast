from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

registry = CollectorRegistry()
g_test = Gauge('test_metric', 'A test metric', registry=registry)
g_test.set(42)

push_to_gateway('http://localhost:9091', job='test_job', registry=registry)
