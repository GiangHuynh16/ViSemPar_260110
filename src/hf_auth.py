"""
HuggingFace Authentication Helper for Training
Handles automatic login before training starts
"""

import os
import sys
from pathlib import Path


def ensure_hf_login(require_write=True):
    """
    Check if user is logged in to HuggingFace (non-blocking)

    Args:
        require_write: Whether to require write permission (for push to hub)

    Returns:
        bool: True if login successful or already logged in, False otherwise

    Note:
        This function does NOT force login or exit on failure.
        Use manual login: huggingface-cli login
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("⚠️  huggingface_hub not installed (optional for training)")
        print("Install with: pip install huggingface_hub")
        return False

    # Check if already logged in (using cached credentials from huggingface-cli)
    try:
        api = HfApi()
        user_info = api.whoami()

        print("✅ Already logged in to HuggingFace")
        print(f"   User: {user_info['name']}")

        return True

    except Exception:
        # Not logged in - this is OK, just inform user
        print("ℹ️  Not logged in to HuggingFace (optional for training)")
        print("\nTo enable model downloads and HF features, login manually:")
        print("  huggingface-cli login")
        print("  # Then paste your token from https://huggingface.co/settings/tokens")
        print("\nContinuing without HuggingFace login...")

        return False


def get_hf_token():
    """
    Get HuggingFace token from various sources

    Priority:
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable
    3. .env file
    4. Return None (will need manual login)

    Returns:
        str or None: Token if found
    """
    # Try environment variables
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

    if token:
        return token

    # Try .env file
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            token = os.getenv('HF_TOKEN')
            if token:
                return token
        except ImportError:
            # python-dotenv not installed, skip
            pass

    return None


def check_hf_token_valid(token):
    """
    Check if HuggingFace token is valid

    Args:
        token: HF token string

    Returns:
        bool: True if valid
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.whoami(token=token)
        return True

    except Exception:
        return False


def get_hf_username():
    """
    Get logged-in HuggingFace username

    Returns:
        str or None: Username if logged in
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user_info = api.whoami()
        return user_info['name']

    except Exception:
        return None


def setup_hf_cache_dir(cache_dir=None):
    """
    Setup HuggingFace cache directory

    Args:
        cache_dir: Custom cache directory (optional)

    Returns:
        Path: Cache directory path
    """
    if cache_dir is None:
        # Use default: ~/.cache/huggingface
        cache_dir = Path.home() / '.cache' / 'huggingface'
    else:
        cache_dir = Path(cache_dir)

    # Create if not exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable
    os.environ['HF_HOME'] = str(cache_dir)

    return cache_dir


if __name__ == "__main__":
    # Test authentication
    print("Testing HuggingFace Authentication...")
    ensure_hf_login()
