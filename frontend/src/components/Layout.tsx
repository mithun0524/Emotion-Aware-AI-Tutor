import Link from 'next/link';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="page-container">
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="brand">
              <h1 className="text-xl font-semibold">AI Tutor</h1>
            </div>
            <nav className="nav">
              <Link href="/" className="nav-link">
                Chat
              </Link>
              <Link href="/voice-chat" className="nav-link">
                Voice Chat
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          {children}
        </div>
      </main>
    </div>
  );
}
