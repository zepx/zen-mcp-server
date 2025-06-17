"""
Tests for the Consensus tool
"""

import json
import unittest
from unittest.mock import Mock, patch

from tools.consensus import ConsensusTool


class TestConsensusTool(unittest.TestCase):
    """Test cases for the Consensus tool"""

    def setUp(self):
        """Set up test fixtures"""
        self.tool = ConsensusTool()

    def test_tool_metadata(self):
        """Test tool metadata is correct"""
        self.assertEqual(self.tool.get_name(), "consensus")
        self.assertTrue("MULTI-MODEL CONSENSUS" in self.tool.get_description())
        self.assertEqual(self.tool.get_default_temperature(), 0.2)

    def test_input_schema(self):
        """Test input schema is properly defined"""
        schema = self.tool.get_input_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("prompt", schema["properties"])
        self.assertIn("models", schema["properties"])
        self.assertEqual(schema["required"], ["prompt", "models"])

        # Check that schema includes stance information
        models_desc = schema["properties"]["models"]["description"]
        # Check that ONLY is emphasized
        self.assertIn("ONLY these stance words are supported", models_desc)
        # Check supportive stances
        self.assertIn("Supportive: 'for', 'support', 'favor'", models_desc)
        self.assertIn("'o3:for'", models_desc)
        self.assertIn("'pro:support'", models_desc)
        self.assertIn("'grok:favor'", models_desc)
        # Check critical stances
        self.assertIn("Critical: 'against', 'oppose', 'critical'", models_desc)
        self.assertIn("'o3:against'", models_desc)
        self.assertIn("'pro:oppose'", models_desc)
        self.assertIn("'grok:critical'", models_desc)
        # Check default guidance
        self.assertIn("Default to neutral unless user requests debate format", models_desc)

    def test_parse_model_and_stance_basic(self):
        """Test basic model and stance parsing"""
        # Test basic stances
        self.assertEqual(self.tool._parse_model_and_stance("o3:for"), ("o3", "for"))
        self.assertEqual(self.tool._parse_model_and_stance("pro:against"), ("pro", "against"))
        self.assertEqual(self.tool._parse_model_and_stance("grok-3"), ("grok-3", "neutral"))

        # Test empty stance
        self.assertEqual(self.tool._parse_model_and_stance("o3:"), ("o3", "neutral"))

        # Test spaces
        self.assertEqual(self.tool._parse_model_and_stance(" o3 : for "), ("o3", "for"))

    def test_parse_model_and_stance_synonyms(self):
        """Test stance synonym parsing"""
        # Supportive synonyms
        self.assertEqual(self.tool._parse_model_and_stance("o3:support"), ("o3", "for"))
        self.assertEqual(self.tool._parse_model_and_stance("o3:favor"), ("o3", "for"))

        # Critical synonyms
        self.assertEqual(self.tool._parse_model_and_stance("pro:critical"), ("pro", "against"))
        self.assertEqual(self.tool._parse_model_and_stance("pro:oppose"), ("pro", "against"))

        # Case insensitive
        self.assertEqual(self.tool._parse_model_and_stance("o3:FOR"), ("o3", "for"))
        self.assertEqual(self.tool._parse_model_and_stance("o3:Support"), ("o3", "for"))
        self.assertEqual(self.tool._parse_model_and_stance("pro:AGAINST"), ("pro", "against"))
        self.assertEqual(self.tool._parse_model_and_stance("pro:Critical"), ("pro", "against"))

        # Test removed synonyms now fail
        result = self.tool._parse_model_and_stance("o3:supportive")
        self.assertIsNone(result[0])
        self.assertIn("invalid stance", result[1])

        result = self.tool._parse_model_and_stance("o3:pro")
        self.assertIsNone(result[0])
        self.assertIn("invalid stance", result[1])

        result = self.tool._parse_model_and_stance("pro:contra")
        self.assertIsNone(result[0])
        self.assertIn("invalid stance", result[1])

        result = self.tool._parse_model_and_stance("pro:con")
        self.assertIsNone(result[0])
        self.assertIn("invalid stance", result[1])

    def test_parse_model_and_stance_errors(self):
        """Test error cases in model and stance parsing"""
        # Empty model name
        result = self.tool._parse_model_and_stance(":for")
        self.assertIsNone(result[0])
        self.assertIn("model name cannot be empty", result[1])

        # Invalid stance
        result = self.tool._parse_model_and_stance("o3:maybe")
        self.assertIsNone(result[0])
        self.assertIn("invalid stance", result[1])
        self.assertIn("maybe", result[1])

        # Empty string
        result = self.tool._parse_model_and_stance("")
        self.assertIsNone(result[0])
        self.assertIn("model name cannot be empty", result[1])

    def test_validate_model_combinations(self):
        """Test model combination validation"""
        # Valid combinations
        valid, skipped = self.tool._validate_model_combinations(["o3:for", "pro:against", "grok-3", "o3:against"])
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(skipped), 0)

        # Test max instances per combination (2)
        valid, skipped = self.tool._validate_model_combinations(
            ["o3:for", "o3:for", "o3:for", "pro:against"]  # This should be skipped
        )
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(skipped), 1)
        self.assertIn("max 2 instances", skipped[0])

        # Test invalid stances
        valid, skipped = self.tool._validate_model_combinations(["o3:maybe", "pro:kinda", "grok-3"])
        self.assertEqual(len(valid), 1)  # Only grok-3 is valid
        self.assertEqual(len(skipped), 2)

    def test_get_stance_enhanced_prompt(self):
        """Test stance-enhanced prompt generation"""
        # Test that stance prompts are injected correctly
        for_prompt = self.tool._get_stance_enhanced_prompt("for")
        self.assertIn("SUPPORTIVE PERSPECTIVE", for_prompt)

        against_prompt = self.tool._get_stance_enhanced_prompt("against")
        self.assertIn("CRITICAL PERSPECTIVE", against_prompt)

        neutral_prompt = self.tool._get_stance_enhanced_prompt("neutral")
        self.assertIn("BALANCED PERSPECTIVE", neutral_prompt)

    def test_format_consensus_output(self):
        """Test consensus output formatting"""
        responses = [
            {"model": "o3", "stance": "for", "status": "success", "verdict": "Good idea"},
            {"model": "pro", "stance": "against", "status": "success", "verdict": "Bad idea"},
            {"model": "grok", "stance": "neutral", "status": "error", "error": "Timeout"},
        ]
        skipped = ["flash:maybe (invalid stance)"]

        output = self.tool._format_consensus_output(responses, skipped)
        output_data = json.loads(output)

        self.assertEqual(output_data["status"], "consensus_success")
        self.assertEqual(output_data["models_used"], ["o3:for", "pro:against"])
        self.assertEqual(output_data["models_skipped"], skipped)
        self.assertEqual(output_data["models_errored"], ["grok"])
        self.assertIn("next_steps", output_data)

    @patch("tools.consensus.ConsensusTool.get_model_provider")
    async def test_execute_with_stance_synonyms(self, mock_get_provider):
        """Test execute with stance synonyms"""
        # Mock provider
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_provider.generate_content.return_value = mock_response
        mock_get_provider.return_value = mock_provider

        # Test with various stance synonyms
        result = await self.tool.execute(
            {"prompt": "Test prompt", "models": ["o3:support", "pro:critical", "grok:favor"]}
        )

        # Verify all models were called
        self.assertEqual(mock_get_provider.call_count, 3)

        # Check that response contains expected format
        response_text = result[0].text
        response_data = json.loads(response_text)
        self.assertEqual(response_data["status"], "consensus_success")
        self.assertEqual(len(response_data["models_used"]), 3)


if __name__ == "__main__":
    unittest.main()
