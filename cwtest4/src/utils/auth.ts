import { Doctor } from '@/types';

export const getStoredUser = (): Doctor | null => {
  if (typeof window === 'undefined') return null;
  const user = localStorage.getItem('user');
  return user ? JSON.parse(user) : null;
};

export const setStoredUser = (user: Doctor): void => {
  if (typeof window === 'undefined') return;
  localStorage.setItem('user', JSON.stringify(user));
};

export const removeStoredUser = (): void => {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('user');
};

export const isAuthenticated = (): boolean => {
  return !!getStoredUser();
}; 