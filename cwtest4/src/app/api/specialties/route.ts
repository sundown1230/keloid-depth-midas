import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { ApiResponse } from '@/types';

const specialties = [
  '内科',
  '外科',
  '小児科',
  '産婦人科',
  '眼科',
  '耳鼻咽喉科',
  '皮膚科',
  '精神科',
  '整形外科',
  '泌尿器科',
  '歯科',
  'その他'
];

export async function GET() {
  try {
    return NextResponse.json<ApiResponse<string[]>>({
      success: true,
      data: specialties
    });
  } catch (error) {
    console.error('Fetch specialties error:', error);
    return NextResponse.json<ApiResponse<null>>({
      success: false,
      error: 'Internal server error'
    }, { status: 500 });
  }
} 