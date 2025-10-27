# prompts/versions.py
"""
Prompt version tracking and changelog.
Allows correlation of agent performance with specific prompt versions.
"""

PROMPT_VERSION = "v2.0"

PROMPT_CHANGELOG = """
v2.0 (2025-10-25) - Major Refactoring:
- Reduced prompt from 350 to 80 lines (-77% reduction)
- Removed all contradicting instructions (permission protocol vs complete workflows)
- Eliminated 20+ manual tool listings (let LangChain auto-inject)
- Replaced "CRITICAL"/"MUST" shouty directives with examples
- Consolidated 15 conflicting guidelines to 5 core principles
- Added ✅/❌ interpretation examples (more effective than rules)
- Simplified ML workflow from 40 lines to 2 lines
- Integrated ToolRegistry for maintainable tool management
- Token savings: ~900 tokens per query (~$0.005-0.01 saved)

v1.0 (Initial) - Organic Growth Phase:
- 350-line monolithic prompt
- Detailed tool descriptions (6-10 lines each)
- Prescriptive workflows with MANDATORY ORDER
- Permission protocol (conflicted with immediate execution)
- 15 priority guidelines (many contradictory)
- Heavy use of CRITICAL/MUST/ALWAYS directives
"""

def get_version_info():
    """Get current prompt version and changelog"""
    return {
        'version': PROMPT_VERSION,
        'changelog': PROMPT_CHANGELOG
    }
