import { useRoute, useLocation } from "wouter";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2, ArrowLeft, FileText, Download } from "lucide-react";
import { Streamdown } from "streamdown";
import { toast } from "sonner";

export default function ProductView() {
  const [, params] = useRoute("/product/:id");
  const [, navigate] = useLocation();
  const productId = params?.id ? parseInt(params.id) : 0;

  const { data, isLoading } = trpc.products.getById.useQuery(
    { id: productId },
    { enabled: productId > 0 }
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
            <Button onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const { product, modules, assets } = data;

  const handleExportModule = (module: any) => {
    // Create a combined document with both on-screen doc and script
    const content = `# ${module.title}

## On-Screen Document

${module.onScreenDoc}

---

## Narration Script

${module.script}

---

**Estimated Duration:** ${module.estimatedDuration} minutes
`;

    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${module.title.replace(/[^a-z0-9]/gi, "_")}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Module exported!");
  };

  const handleExportAsset = (asset: any) => {
    const blob = new Blob([asset.content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${asset.title.replace(/[^a-z0-9]/gi, "_")}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Asset exported!");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 py-8">
      <div className="container max-w-6xl">
        <Button variant="ghost" onClick={() => navigate("/dashboard")} className="mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>

        <div className="mb-8">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold">{product.title}</h1>
              <p className="text-gray-600 mt-2">{product.description}</p>
            </div>
            <Badge
              variant={
                product.status === "published"
                  ? "default"
                  : product.status === "draft"
                  ? "secondary"
                  : "outline"
              }
            >
              {product.status}
            </Badge>
          </div>

          <div className="grid grid-cols-3 gap-4 mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Niche</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="font-medium">{product.niche}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Target Audience</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="font-medium">{product.targetAudience}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Modules</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="font-medium">{modules.length} modules</p>
              </CardContent>
            </Card>
          </div>
        </div>

        <Tabs defaultValue="modules" className="space-y-4">
          <TabsList>
            <TabsTrigger value="modules">Course Modules</TabsTrigger>
            <TabsTrigger value="assets">Assets & Frameworks</TabsTrigger>
          </TabsList>

          <TabsContent value="modules" className="space-y-4">
            {modules.length === 0 ? (
              <Card>
                <CardContent className="py-8 text-center text-gray-500">
                  No modules generated yet
                </CardContent>
              </Card>
            ) : (
              modules.map((module, index) => (
                <Card key={module.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle>
                          Module {index + 1}: {module.title}
                        </CardTitle>
                        <CardDescription>{module.description}</CardDescription>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleExportModule(module)}
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Export
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="doc">
                      <TabsList>
                        <TabsTrigger value="doc">On-Screen Document</TabsTrigger>
                        <TabsTrigger value="script">Narration Script</TabsTrigger>
                      </TabsList>

                      <TabsContent value="doc" className="mt-4">
                        <div className="prose max-w-none bg-white p-6 rounded-lg border">
                          <Streamdown>{module.onScreenDoc || "No content"}</Streamdown>
                        </div>
                      </TabsContent>

                      <TabsContent value="script" className="mt-4">
                        <div className="bg-white p-6 rounded-lg border">
                          <pre className="whitespace-pre-wrap font-sans text-sm">
                            {module.script || "No script"}
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
                  No assets generated yet
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {assets.map((asset) => (
                  <Card key={asset.id}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-lg">{asset.title}</CardTitle>
                          <CardDescription>
                            <Badge variant="outline" className="mt-1">
                              {asset.type}
                            </Badge>
                          </CardDescription>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleExportAsset(asset)}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      </div>
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
