import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { ApiResponse, Doctor } from '@/types';

export async function GET(request: NextRequest) {
  try {
    const token = request.headers.get('Authorization')?.replace('Bearer ', '');

    if (!token) {
      return NextResponse.json<ApiResponse<null>>({
        success: false,
        error: 'Unauthorized'
      }, { status: 401 });
    }

    // Call external API
    const response = await fetch(`${process.env.API_URL}/api/doctors/me`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json<ApiResponse<null>>({
        success: false,
        error: data.error || 'Failed to fetch user data'
      }, { status: response.status });
    }

    return NextResponse.json<ApiResponse<Doctor>>({
      success: true,
      data: data
    });
  } catch (error) {
    console.error('Fetch user data error:', error);
    return NextResponse.json<ApiResponse<null>>({
      success: false,
      error: 'Internal server error'
    }, { status: 500 });
  }
} 