import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { LoginRequest, ApiResponse, Doctor } from '@/types';

export async function POST(request: NextRequest) {
  try {
    const body: LoginRequest = await request.json();
    const { email, password } = body;

    // Validate input
    if (!email || !password) {
      return NextResponse.json<ApiResponse<null>>({
        success: false,
        error: 'Email and password are required'
      }, { status: 400 });
    }

    // Call external API
    const response = await fetch(`${process.env.API_URL}/api/doctors/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json<ApiResponse<null>>({
        success: false,
        error: data.error || 'Login failed'
      }, { status: response.status });
    }

    return NextResponse.json<ApiResponse<Doctor>>({
      success: true,
      data: data
    });
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json<ApiResponse<null>>({
      success: false,
      error: 'Internal server error'
    }, { status: 500 });
  }
} 