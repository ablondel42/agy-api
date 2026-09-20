# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from integrations.google_adk.google_adk_integration import create_agent

logger = logging.getLogger(__name__)

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    return {
        "status": "success",
        "report": (
            f"The weather in {city} is sunny with a temperature of 25 degrees"
            f" Celsius (77 degrees Fahrenheit)."
        ),
    }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city using the local system timezone.

    Args: city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    now = datetime.datetime.now().astimezone()
    tz_name = now.tzname()  # e.g., 'CEST', 'EDT', or 'UTC'
    report = f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})."
    return {
        "status": "success",
        "timezone": tz_name,
        "report": report,
    }


logger.info("Configuring root_agent 'weather_time_agent'")
root_agent = create_agent(
    name="weather_time_agent",
    model="nvidia/nemotron-3-ultra-550b-a55b",
    description=(
        "Agent to answer questions about the time, climate, and weather in cities worldwide."
    ),
    instruction=(
        "You are an intelligent assistant that answers user questions about cities, "
        "including climate, weather patterns, timezones, and geographical context. "
        "Use your full reasoning and inference capabilities to provide accurate, "
        "helpful, and detailed answers."
    ),
    tools=[get_weather, get_current_time]
)
logger.info("root_agent 'weather_time_agent' created successfully")