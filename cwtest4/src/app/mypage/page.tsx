'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Doctor, ApiResponse } from '@/types';
import { getStoredUser, removeStoredUser } from '@/utils/auth';

export default function MyPage() {
  const router = useRouter();
  const [user, setUser] = useState<Doctor | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const storedUser = getStoredUser();
    if (!storedUser) {
      router.push('/login');
      return;
    }

    const fetchUserData = async () => {
      try {
        const response = await fetch('/api/me', {
          headers: {
            'Authorization': `Bearer ${storedUser.id}`,
          },
        });

        const data: ApiResponse<Doctor> = await response.json();

        if (data.success && data.data) {
          setUser(data.data);
        } else {
          setError(data.error || 'ユーザー情報の取得に失敗しました');
        }
      } catch (error) {
        setError('ユーザー情報の取得中にエラーが発生しました');
      }
    };

    fetchUserData();
  }, [router]);

  const handleLogout = () => {
    removeStoredUser();
    router.push('/login');
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-md">
          {error ? (
            <div className="text-red-600">{error}</div>
          ) : (
            <div>読み込み中...</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 className="text-2xl font-bold text-center mb-8">
          マイページ
        </h1>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">名前</label>
            <div className="mt-1 p-2 bg-gray-50 rounded">{user.name}</div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">メールアドレス</label>
            <div className="mt-1 p-2 bg-gray-50 rounded">{user.email}</div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">専門分野</label>
            <div className="mt-1 p-2 bg-gray-50 rounded">{user.specialty}</div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">医師免許番号</label>
            <div className="mt-1 p-2 bg-gray-50 rounded">{user.licenseNumber}</div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full bg-red-500 text-white py-2 px-4 rounded hover:bg-red-600 transition-colors"
          >
            ログアウト
          </button>
        </div>
      </div>
    </div>
  );
} 