#!/usr/bin/env python3
"""
Health check script for RAG System
Run this to diagnose system issues quickly
"""

import os
import anthropic
from config import config
from vector_store import VectorStore


def check_environment():
    """Check environment configuration"""
    print("=== ENVIRONMENT CHECK ===")
    
    # Check .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print("✓ .env file exists")
    else:
        print("✗ .env file missing")
        return False
    
    # Check API key is configured
    if config.ANTHROPIC_API_KEY:
        print(f"✓ ANTHROPIC_API_KEY configured ({len(config.ANTHROPIC_API_KEY)} characters)")
    else:
        print("✗ ANTHROPIC_API_KEY not configured")
        return False
    
    print(f"✓ Model: {config.ANTHROPIC_MODEL}")
    print(f"✓ ChromaDB path: {config.CHROMA_PATH}")
    return True


def check_anthropic_api():
    """Check Anthropic API connectivity"""
    print("\n=== ANTHROPIC API CHECK ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("✗ No API key to test")
        return False
    
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": "Test connection"}]
        )
        print("✓ Anthropic API connection successful")
        print(f"✓ Response: {response.content[0].text[:50]}...")
        return True
    
    except anthropic.AuthenticationError as e:
        print(f"✗ AUTHENTICATION ERROR: {e}")
        print("  → API key is invalid or expired")
        print("  → Get a new key from https://console.anthropic.com/")
        return False
    
    except anthropic.APIError as e:
        print(f"✗ API ERROR: {e}")
        return False
    
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        return False


def check_vector_database():
    """Check vector database health"""
    print("\n=== VECTOR DATABASE CHECK ===")
    
    try:
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Check course count
        course_count = store.get_course_count()
        if course_count > 0:
            print(f"✓ {course_count} courses loaded")
            
            # List courses
            course_titles = store.get_existing_course_titles()
            for title in course_titles:
                print(f"  - {title}")
            
            # Test search
            results = store.search("introduction")
            if not results.is_empty():
                print(f"✓ Vector search functional ({len(results.documents)} results)")
                return True
            else:
                print("✗ Vector search returns no results")
                return False
        else:
            print("✗ No courses found in database")
            print("  → Run the application once to load course documents")
            return False
            
    except Exception as e:
        print(f"✗ Vector database error: {e}")
        return False


def main():
    """Run complete health check"""
    print("RAG SYSTEM HEALTH CHECK")
    print("=" * 50)
    
    checks = [
        ("Environment", check_environment),
        ("Vector Database", check_vector_database), 
        ("Anthropic API", check_anthropic_api),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - System is healthy!")
    else:
        print("⚠️  ISSUES FOUND - See details above")
        print("\nMost common fix:")
        print("1. Update ANTHROPIC_API_KEY in .env file")
        print("2. Restart the application")
    
    return all_passed


if __name__ == "__main__":
    main()