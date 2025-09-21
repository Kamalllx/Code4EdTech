#!/usr/bin/env python3
"""
Test API Endpoints
Quick test to verify all API endpoints are working
"""

import requests
import json
import time

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing API Endpoints")
    print("=" * 50)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    endpoints_to_test = [
        ("/api/health", "Health Check"),
        ("/api/students", "Get Students"),
        ("/api/exams", "Get Exams"),
        ("/api/statistics/overview", "Overview Statistics"),
        ("/api/recent-activity", "Recent Activity"),
    ]
    
    results = []
    
    for endpoint, description in endpoints_to_test:
        try:
            print(f"\n🔍 Testing {description}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {description}: SUCCESS")
                data = response.json()
                if isinstance(data, list):
                    print(f"   📊 Returned {len(data)} items")
                elif isinstance(data, dict):
                    print(f"   📊 Returned {len(data)} fields")
                results.append(True)
            else:
                print(f"❌ {description}: FAILED (Status: {response.status_code})")
                results.append(False)
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {description}: CONNECTION ERROR")
            results.append(False)
        except Exception as e:
            print(f"❌ {description}: ERROR - {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All API endpoints are working correctly!")
        print("🌐 Frontend should now be able to connect successfully.")
    else:
        print(f"\n⚠️  {total - passed} endpoint(s) failed. Check the backend logs.")
    
    return passed == total

if __name__ == "__main__":
    test_api_endpoints()
