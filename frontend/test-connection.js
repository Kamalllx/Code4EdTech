// Simple test script to verify frontend-backend connection
const axios = require('axios');

const API_BASE_URL = 'http://localhost:5000';

async function testConnection() {
  console.log('🧪 Testing Frontend-Backend Connection...\n');
  
  try {
    // Test health endpoint
    console.log('1. Testing health endpoint...');
    const healthResponse = await axios.get(`${API_BASE_URL}/api/health`);
    console.log('✅ Health check passed:', healthResponse.data);
  } catch (error) {
    console.log('❌ Health check failed:', error.message);
    console.log('   Make sure your backend is running on http://localhost:5000');
  }
  
  try {
    // Test students endpoint
    console.log('\n2. Testing students endpoint...');
    const studentsResponse = await axios.get(`${API_BASE_URL}/api/students`);
    console.log('✅ Students endpoint working:', studentsResponse.data);
  } catch (error) {
    console.log('❌ Students endpoint failed:', error.message);
  }
  
  try {
    // Test exams endpoint
    console.log('\n3. Testing exams endpoint...');
    const examsResponse = await axios.get(`${API_BASE_URL}/api/exams`);
    console.log('✅ Exams endpoint working:', examsResponse.data);
  } catch (error) {
    console.log('❌ Exams endpoint failed:', error.message);
  }
  
  console.log('\n📋 Summary:');
  console.log('- Frontend: http://localhost:3000');
  console.log('- Backend: http://localhost:5000');
  console.log('- Make sure both servers are running');
  console.log('- Check the API_REQUIREMENTS.md for endpoint specifications');
}

testConnection().catch(console.error);
