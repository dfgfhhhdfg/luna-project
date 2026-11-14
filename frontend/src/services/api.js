/**
 * LUNA NAVIGATOR — Frontend API Service Client (Member 1 Integration)
 * Matches shared REST contracts in docs/api-contracts.md
 */

import axios from 'axios';

const API_BASE = '/api';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// API Error interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[Luna API Error]', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// 1. Candidate Sites
export const getSites = async () => {
  const response = await apiClient.get('/sites');
  return response.data;
};

export const getSite = async (siteId) => {
  const response = await apiClient.get(`/sites/${siteId}`);
  return response.data;
};

// 2. Ephemeris Engine
export const calculateEphemeris = async (params) => {
  const response = await apiClient.post('/ephemeris', params);
  return response.data;
};

// 3. Terrain Horizon Engine
export const getTerrainHorizon = async (params) => {
  const response = await apiClient.post('/terrain/horizon', params);
  return response.data;
};

export const getSolarObstruction = async (params) => {
  const response = await apiClient.post('/terrain/solar-obstruction', params);
  return response.data;
};

export const getEarthObstruction = async (params) => {
  const response = await apiClient.post('/terrain/earth-obstruction', params);
  return response.data;
};

export const getSurfaceViewshed = async (params) => {
  const response = await apiClient.post('/terrain/viewshed', params);
  return response.data;
};

export const getTerrainSample = async (latitude, longitude) => {
  const response = await apiClient.get('/terrain/sample', { params: { latitude, longitude } });
  return response.data;
};

export const getTerrainProfile = async (params) => {
  const response = await apiClient.post('/terrain/profile', params);
  return response.data;
};

export const assessLandingArea = async (params) => {
  const response = await apiClient.post('/terrain/assess-landing-area', params);
  return response.data;
};

export const sampleTerrainRay = async (params) => {
  const response = await apiClient.post('/terrain/ray', params);
  return response.data;
};

export const checkLineOfSight = async (params) => {
  const response = await apiClient.post('/terrain/check-los', params);
  return response.data;
};

export const getDemHorizonProfile = async (params) => {
  const response = await apiClient.post('/terrain/dem-horizon', params);
  return response.data;
};

// 4. Solar Illumination Engine
export const calculateIllumination = async (params) => {
  const response = await apiClient.post('/illumination', params);
  return response.data;
};

// 5. Earth Communication Engine
export const calculateCommunication = async (params) => {
  const response = await apiClient.post('/communication', params);
  return response.data;
};

// 6. Site Comparison
export const compareSites = async (params) => {
  const response = await apiClient.post('/compare', params);
  return response.data;
};

// 7. Site Recommendation & MCDA
export const recommendSites = async (params) => {
  const response = await apiClient.post('/recommend', params);
  return response.data;
};

// 8. Mission Energy Simulation
export const simulateMission = async (params) => {
  const response = await apiClient.post('/simulate', params);
  return response.data;
};

// Health Check
export const checkHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};
