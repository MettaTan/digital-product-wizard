import { useRoute, useLocation } from "wouter";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, ArrowLeft, Lock } from "lucide-react";
import { Streamdown } from "streamdown";
import { useAuth } from "@/_core/hooks/useAuth";

export default function MyProduct() {
  const [, params] = useRoute("/my-products/:id");
  const [, navigate] = useLocation();
  const { isAuthenticated } = useAuth();
  const productId = params?.id ? parseInt(params.id) : 0;

  const { data: purchases } = trpc.products.myPurchases.useQuery(undefined, {
    enabled: isAuthenticated,
  });

  const { data, isLoading } = trpc.products.getById.useQuery(
    { id: productId },
    { enabled: productId > 0 }
  );

  const hasPurchased = purchases?.some(
    (p) => p.productId === productId && p.status === "active"
  );

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-purple-600" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card>
          <CardHeader>
            <CardTitle>Product Not Found</CardTitle>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate("/marketplace")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Marketplace
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!hasPurchased) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-white to-blue-50">
        <Card className="max-w-md">
          <CardHeader>
            <div className="flex justify-center mb-4">
              <Lock className="h-12 w-12 text-gray-400" />
            </div>
            <CardTitle className="text-center">Access Denied</CardTitle>
            <CardDescription className="text-center">
              You need to purchase this product to access its content
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate("/marketplace")} className="w-full">
              Go to Marketplace
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const { product, modules, assets } = data;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 py-8">
      <div className="container max-w-6xl">
        <Button variant="ghost" onClick={() => navigate("/marketplace")} className="mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Marketplace
        </Button>

        <div className="mb-8">
          <h1 className="text-4xl font-bold">{product.title}</h1>
          <p className="text-gray-600 mt-2">{product.description}</p>
        </div>

        <Tabs defaultValue="modules" className="space-y-4">
          <TabsList>
            <TabsTrigger value="modules">Course Modules</TabsTrigger>
            <TabsTrigger value="assets">Resources & Assets</TabsTrigger>
          </TabsList>

          <TabsContent value="modules" className="space-y-4">
            {modules.length === 0 ? (
              <Card>
                <CardContent className="py-8 text-center text-gray-500">
                  No modules available
                </CardContent>
              </Card>
            ) : (
              modules.map((module, index) => (
                <Card key={module.id}>
                  <CardHeader>
                    <CardTitle>
                      Module {index + 1}: {module.title}
                    </CardTitle>
                    <CardDescription>{module.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="doc">
                      <TabsList>
                        <TabsTrigger value="doc">Course Content</TabsTrigger>
                        <TabsTrigger value="script">Teaching Notes</TabsTrigger>
                      </TabsList>

                      <TabsContent value="doc" className="mt-4">
                        <div className="prose max-w-none bg-white p-6 rounded-lg border">
                          <Streamdown>{module.onScreenDoc || "No content"}</Streamdown>
                        </div>
                      </TabsContent>

                      <TabsContent value="script" className="mt-4">
                        <div className="bg-white p-6 rounded-lg border">
                          <pre className="whitespace-pre-wrap font-sans text-sm">
                            {module.script || "No teaching notes"}
                          </pre>
                        </div>
                      </TabsContent>
                    </Tabs>

                    {module.estimatedDuration && (
                      <p className="text-sm text-gray-500 mt-4">
                        Estimated duration: {module.estimatedDuration} minutes
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>

          <TabsContent value="assets" className="space-y-4">
            {assets.length === 0 ? (
              <Card>
                <CardContent className="py-8 text-center text-gray-500">
                  No assets available
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {assets.map((asset) => (
                  <Card key={asset.id}>
                    <CardHeader>
                      <CardTitle className="text-lg">{asset.title}</CardTitle>
                      <CardDescription>{asset.type}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="prose prose-sm max-w-none bg-gray-50 p-4 rounded border max-h-64 overflow-y-auto">
                        <Streamdown>{asset.content || "No content"}</Streamdown>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
