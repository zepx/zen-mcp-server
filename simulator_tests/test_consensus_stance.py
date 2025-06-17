"""
Test consensus tool with explicit stance arguments
"""

import json

from .base_test import BaseSimulatorTest


class TestConsensusStance(BaseSimulatorTest):
    """Test consensus tool functionality with stance steering"""

    @property
    def test_name(self) -> str:
        return "consensus_stance"

    @property
    def test_description(self) -> str:
        return "Test consensus tool with stance steering (for/against/neutral)"

    def run_test(self) -> bool:
        """Run consensus stance test"""
        try:
            self.logger.info("Testing consensus tool with explicit flash:for and flash:against stances")

            # Send request with explicit stances as user requested
            response, continuation_id = self.call_mcp_tool(
                "consensus",
                {
                    "prompt": "How about I add a button to order pizza directly to my log_analyzer app?",
                    "models": ["flash:for", "flash:against"],
                    "model": "flash",  # Default model for Claude's synthesis
                },
            )

            # Validate response
            if not response:
                self.logger.error("Failed to get response from consensus tool")
                return False

            self.logger.info(f"Consensus response preview: {response[:500]}...")

            # Parse the JSON response
            try:
                consensus_data = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse consensus response as JSON: {response}")
                return False

            # Validate consensus structure
            if "status" not in consensus_data:
                self.logger.error("Missing 'status' field in consensus response")
                return False

            if consensus_data["status"] != "consensus_success":
                self.logger.error(f"Consensus failed with status: {consensus_data['status']}")
                return False

            # Check that both models were used with their stances
            if "models_used" not in consensus_data:
                self.logger.error("Missing 'models_used' field in consensus response")
                return False

            models_used = consensus_data["models_used"]
            if len(models_used) != 2:
                self.logger.error(f"Expected 2 models, got {len(models_used)}")
                return False

            if "flash:for" not in models_used:
                self.logger.error("Missing 'flash:for' in models_used")
                return False

            if "flash:against" not in models_used:
                self.logger.error("Missing 'flash:against' in models_used")
                return False

            # Validate responses structure
            if "responses" not in consensus_data:
                self.logger.error("Missing 'responses' field in consensus response")
                return False

            responses = consensus_data["responses"]
            if len(responses) != 2:
                self.logger.error(f"Expected 2 responses, got {len(responses)}")
                return False

            # Check each response has the correct stance
            for_response = None
            against_response = None

            for resp in responses:
                if "stance" not in resp:
                    self.logger.error("Missing 'stance' field in response")
                    return False

                if resp["stance"] == "for":
                    for_response = resp
                elif resp["stance"] == "against":
                    against_response = resp

            # Verify we got both stances
            if not for_response:
                self.logger.error("Missing 'for' stance response")
                return False

            if not against_response:
                self.logger.error("Missing 'against' stance response")
                return False

            # Check that successful responses have verdicts
            if for_response.get("status") == "success":
                if "verdict" not in for_response:
                    self.logger.error("Missing 'verdict' in for_response")
                    return False
                self.logger.info(f"FOR stance verdict preview: {for_response['verdict'][:200]}...")

            if against_response.get("status") == "success":
                if "verdict" not in against_response:
                    self.logger.error("Missing 'verdict' in against_response")
                    return False
                self.logger.info(f"AGAINST stance verdict preview: {against_response['verdict'][:200]}...")

            # Verify synthesis guidance is present
            if "next_steps" not in consensus_data:
                self.logger.error("Missing 'next_steps' field in consensus response")
                return False

            self.logger.info("✓ Consensus tool correctly processed explicit stance arguments")

            # Test with stance synonyms
            self.logger.info("\nTesting consensus tool with stance synonyms...")
            response2, _ = self.call_mcp_tool(
                "consensus",
                {
                    "prompt": "Should we implement a real-time notification system?",
                    "models": ["o3:support", "pro:oppose"],
                    "model": "flash",
                },
            )

            if not response2:
                self.logger.error("Failed to get response for synonym test")
                return False

            try:
                consensus_data2 = json.loads(response2)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse synonym test response as JSON: {response2}")
                return False

            # Check that synonyms were normalized
            models_used2 = consensus_data2.get("models_used", [])
            if "o3:for" not in models_used2:  # support -> for
                self.logger.error("Failed to normalize 'support' to 'for'")
                return False
            if "pro:against" not in models_used2:  # oppose -> against
                self.logger.error("Failed to normalize 'oppose' to 'against'")
                return False

            self.logger.info("✓ Consensus tool correctly normalized stance synonyms")

            return True

        except Exception as e:
            self.logger.error(f"Test failed with exception: {str(e)}")
            return False
