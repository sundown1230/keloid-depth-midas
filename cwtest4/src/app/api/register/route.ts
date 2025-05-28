import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { RegisterRequest, ApiResponse, Doctor } from '@/types';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // ここで登録処理を行う
    // 例: 外部APIを呼び出す
    const response = await fetch(`${process.env.API_URL}/api/doctors/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json(
      { error: '登録に失敗しました' },
      { status: 500 }
    );
  }
}

// GETメソッドは許可しない
export async function GET() {
  return NextResponse.json(
    { error: 'Method not allowed' },
    { status: 405 }
  );
} 