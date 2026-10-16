import pytest

from cats.analyzer import ContextAnalyzer
from cats.index import LSAEmbedder, ToolIndex
from cats.pipeline import CATS, CATSConfig
from cats.registry import Registry
from cats.tokens import TokenCounter
from cats.types import Branch, Task, Toolkit


def _obj(**props):
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props)[:1]}


CORPUS = [
    (Branch(id="weather", name="Weather", summary="Forecasts, current conditions and severe weather alerts."), [
        (Toolkit(id="openweather", name="OpenWeather", branch_id="weather", summary="Weather data API."), [
            ("current_weather", "Get the current temperature, wind and conditions for a city.", _obj(city="City name")),
            ("forecast", "Get a multi-day weather forecast with rain probability for a location.", _obj(location="Place")),
            ("weather_alerts", "List active severe weather warnings such as storms or floods.", _obj(region="Region")),
        ]),
    ]),
    (Branch(id="chat", name="Team chat", summary="Send and read messages in team chat channels."), [
        (Toolkit(id="slack", name="Slack", branch_id="chat", summary="Slack workspace messaging."), [
            ("post_message", "Post a message to a Slack channel.", _obj(channel="Channel", text="Message text")),
            ("search_messages", "Search Slack message history.", _obj(query="Search terms")),
        ]),
        (Toolkit(id="teams", name="Microsoft Teams", branch_id="chat", summary="Teams channel messaging."), [
            ("send_channel_message", "Send a message to a Microsoft Teams channel.", _obj(team="Team", text="Text")),
        ]),
    ]),
    (Branch(id="code", name="Code execution", summary="Run code, shell commands and tests in a sandbox."), [
        (Toolkit(id="sandbox", name="Python sandbox", branch_id="code", summary="Isolated Python runtime."), [
            ("run_python", "Execute a Python snippet and return stdout.", _obj(code="Python source")),
            ("run_tests", "Run the pytest suite in a directory and report failures.", _obj(path="Directory")),
            ("install_package", "Install a pip package into the sandbox.", _obj(name="Package name")),
        ]),
    ]),
]

QUERIES = [
    ("weather", ["openweather.forecast"], "will it rain in paris tomorrow"),
    ("weather", ["openweather.forecast"], "forecast for berlin this weekend"),
    ("weather", ["openweather.current_weather"], "how hot is it in madrid right now"),
    ("weather", ["openweather.current_weather"], "current temperature in oslo"),
    ("weather", ["openweather.weather_alerts"], "any storm warnings for florida"),
    ("weather", ["openweather.weather_alerts"], "are there flood alerts near the coast"),
    ("chat", ["slack.post_message", "teams.send_channel_message"], "tell the team the deploy is done"),
    ("chat", ["slack.post_message"], "post in the slack channel that lunch is here"),
    ("chat", ["slack.search_messages"], "find the slack thread where we discussed pricing"),
    ("chat", ["teams.send_channel_message"], "send a teams message to the design team"),
    ("chat", ["slack.search_messages"], "search messages about the outage"),
    ("chat", ["slack.post_message", "teams.send_channel_message"], "let everyone in the channel know I'm late"),
    ("code", ["sandbox.run_python"], "run this python snippet print(2+2)"),
    ("code", ["sandbox.run_python"], "execute some code to compute primes"),
    ("code", ["sandbox.run_tests"], "do my unit tests pass"),
    ("code", ["sandbox.run_tests"], "run pytest on the repo"),
    ("code", ["sandbox.install_package"], "install numpy please"),
    ("code", ["sandbox.install_package"], "pip install requests in the sandbox"),
]


@pytest.fixture
def counter():
    return TokenCounter("approx")


@pytest.fixture
def registry(counter):
    reg = Registry(counter)
    for branch, toolkits in CORPUS:
        reg.add_branch(branch)
        for tk, tools in toolkits:
            reg.upsert_toolkit(tk, [{"name": n, "description": d, "input_schema": s} for n, d, s in tools])
    return reg


@pytest.fixture
def tasks():
    return [Task(id=f"t{i}", query=q, gold=[gold], branches=[b], kind="literal") for i, (b, gold, q) in enumerate(QUERIES)]


@pytest.fixture
def index(registry):
    return ToolIndex(registry, LSAEmbedder(dim=32)).build()


@pytest.fixture
def cats(registry, index, tasks):
    c = CATS(registry, index, ContextAnalyzer(registry), CATSConfig(alpha_branch=0.2, alpha_tool=0.2, k_max=3))
    c.fit(tasks)
    c.calibrate(tasks)
    return c
