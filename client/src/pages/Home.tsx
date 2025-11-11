import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { APP_LOGO, APP_TITLE, getLoginUrl } from "@/const";
import { Streamdown } from 'streamdown';

/**
 * All content in this page are only for example, replace with your own feature implementation
 * When building pages, remember your instructions in Frontend Workflow, Frontend Best Practices, Design Guide and Common Pitfalls
 */
export default function Home() {
  const { user, loading, isAuthenticated } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-purple-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
      {/* Hero Section */}
      <div className="container py-20">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-6">
            {APP_TITLE}
          </h1>
          <p className="text-2xl text-gray-700 mb-4">
            Your AI-powered assistant for turning ideas into income
          </p>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-12">
            Create and sell complete digital product packages in minutes — no tech skills needed.
            Generate course modules with scripts, assets, frameworks, and everything you need to launch.
          </p>

          <div className="flex gap-4 justify-center">
            {isAuthenticated ? (
              <>
                <Button asChild size="lg" className="text-lg px-8 py-6">
                  <a href="/dashboard">Go to Dashboard</a>
                </Button>
                <Button asChild size="lg" variant="outline" className="text-lg px-8 py-6">
                  <a href="/marketplace">Browse Marketplace</a>
                </Button>
              </>
            ) : (
              <>
                <Button asChild size="lg" className="text-lg px-8 py-6">
                  <a href={getLoginUrl()}>Get Started Free</a>
                </Button>
                <Button asChild size="lg" variant="outline" className="text-lg px-8 py-6">
                  <a href="/marketplace">Browse Products</a>
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mt-20 max-w-5xl mx-auto">
          <div className="bg-white p-8 rounded-xl shadow-sm">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-2">AI-Powered Generation</h3>
            <p className="text-gray-600">
              Generate complete course outlines, module content with on-screen docs and narration scripts, plus assets and frameworks
            </p>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-sm">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-2">Flexible Pricing</h3>
            <p className="text-gray-600">
              Offer monthly, yearly, or lifetime plans. Automated payments and access control with Stripe integration
            </p>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-sm">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-2">Launch in Minutes</h3>
            <p className="text-gray-600">
              From idea to published product in minutes. Export modules, manage customers, and deliver premium experiences
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
