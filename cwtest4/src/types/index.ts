export interface Doctor {
  id: number;
  name: string;
  email: string;
  specialty: string;
  licenseNumber: string;
  createdAt: string;
  updatedAt: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
  specialty: string;
  licenseNumber: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
} 