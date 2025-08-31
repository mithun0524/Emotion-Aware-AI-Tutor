import React, { PropsWithChildren } from 'react';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'], display: 'swap' });

export default function Layout({ children }: PropsWithChildren) {
  return (
    <div className={inter.className}>
      <main>{children}</main>
    </div>
  );
}
