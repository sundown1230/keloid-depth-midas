import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 className="text-2xl font-bold text-center mb-8">
          医師登録システム
        </h1>
        <div className="space-y-4">
          <Link
            href="/register"
            className="block w-full text-center bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition-colors"
          >
            新規会員登録
          </Link>
          <Link
            href="/login"
            className="block w-full text-center bg-gray-500 text-white py-2 px-4 rounded hover:bg-gray-600 transition-colors"
          >
            ログイン
          </Link>
        </div>
      </div>
    </div>
  );
} 