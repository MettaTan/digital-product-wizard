import { useState } from "react";
import { useLocation } from "wouter";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, CheckCircle2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Streamdown } from "streamdown";

type WizardStep = "details" | "outline" | "modules" | "assets" | "pricing" | "review";

export default function CreateProduct() {
  const [, navigate] = useLocation();
  const [currentStep, setCurrentStep] = useState<WizardStep>("details");
  const [productId, setProductId] = useState<number | null>(null);

  // Step 1: Product Details
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [niche, setNiche] = useState("");
  const [targetAudience, setTargetAudience] = useState("");

  // Step 2: Outline
  const [outline, setOutline] = useState<any>(null);

  // Step 3: Modules (tracked by backend)
  const [modulesGenerated, setModulesGenerated] = useState(false);

  // Step 4: Assets (tracked by backend)
  const [assetsGenerated, setAssetsGenerated] = useState(false);

  // Step 5: Pricing
  const [priceMonthly, setPriceMonthly] = useState("");
  const [priceYearly, setPriceYearly] = useState("");
  const [priceLifetime, setPriceLifetime] = useState("");

  const createProductMutation = trpc.products.create.useMutation();
  const generateOutlineMutation = trpc.products.generateOutline.useMutation();
  const generateModulesMutation = trpc.products.generateModules.useMutation();
  const generateAssetsMutation = trpc.products.generateAssets.useMutation();
  const updateProductMutation = trpc.products.update.useMutation();

  const handleCreateProduct = async () => {
    if (!title || !niche || !targetAudience) {
      toast.error("Please fill in all required fields");
      return;
    }

    try {
      const result = await createProductMutation.mutateAsync({
        title,
        description,
        niche,
        targetAudience,
      });

      setProductId(result.productId);
      toast.success("Product created! Generating course outline...");
      setCurrentStep("outline");

      // Auto-generate outline
      const outlineResult = await generateOutlineMutation.mutateAsync({
        productId: result.productId,
      });

      setOutline(outlineResult);
      toast.success(`Course outline generated with ${outlineResult.modules.length} modules!`);
    } catch (error: any) {
      toast.error(error.message || "Failed to create product");
    }
  };

  const handleGenerateModules = async () => {
    if (!productId) return;

    try {
      toast.info("Generating module content... This may take a minute.");
      const result = await generateModulesMutation.mutateAsync({ productId });
      setModulesGenerated(true);
      toast.success(`Generated ${result.count} modules with scripts and on-screen docs!`);
      setCurrentStep("assets");
    } catch (error: any) {
      toast.error(error.message || "Failed to generate modules");
    }
  };

  const handleGenerateAssets = async () => {
    if (!productId) return;

    try {
      toast.info("Generating assets and frameworks...");
      const result = await generateAssetsMutation.mutateAsync({ productId });
      setAssetsGenerated(true);
      toast.success(`Generated ${result.count} assets!`);
      setCurrentStep("pricing");
    } catch (error: any) {
      toast.error(error.message || "Failed to generate assets");
    }
  };

  const handleSavePricing = async () => {
    if (!productId) return;

    try {
      await updateProductMutation.mutateAsync({
        id: productId,
        priceMonthly: priceMonthly ? Math.round(parseFloat(priceMonthly) * 100) : undefined,
        priceYearly: priceYearly ? Math.round(parseFloat(priceYearly) * 100) : undefined,
        priceLifetime: priceLifetime ? Math.round(parseFloat(priceLifetime) * 100) : undefined,
      });

      toast.success("Pricing saved!");
      setCurrentStep("review");
    } catch (error: any) {
      toast.error(error.message || "Failed to save pricing");
    }
  };

  const handlePublish = async () => {
    if (!productId) return;

    try {
      await updateProductMutation.mutateAsync({
        id: productId,
        status: "published",
      });

      toast.success("Product published! 🎉");
      navigate("/dashboard");
    } catch (error: any) {
      toast.error(error.message || "Failed to publish product");
    }
  };

  const steps = [
    { id: "details", label: "Product Details" },
    { id: "outline", label: "Course Outline" },
    { id: "modules", label: "Generate Modules" },
    { id: "assets", label: "Generate Assets" },
    { id: "pricing", label: "Pricing" },
    { id: "review", label: "Review & Publish" },
  ];

  const currentStepIndex = steps.findIndex((s) => s.id === currentStep);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 py-12">
      <div className="container max-w-4xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            Create Digital Product
          </h1>
          <p className="text-gray-600 mt-2">
            AI-powered wizard to create complete course packages with modules, scripts, and assets
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div
                  className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                    index <= currentStepIndex
                      ? "bg-purple-600 border-purple-600 text-white"
                      : "bg-white border-gray-300 text-gray-400"
                  }`}
                >
                  {index < currentStepIndex ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-0.5 w-12 mx-2 ${
                      index < currentStepIndex ? "bg-purple-600" : "bg-gray-300"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between mt-2">
            {steps.map((step) => (
              <span
                key={step.id}
                className="text-xs text-gray-600 w-24 text-center"
              >
                {step.label}
              </span>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <Card>
          <CardHeader>
            <CardTitle>{steps[currentStepIndex].label}</CardTitle>
          </CardHeader>
          <CardContent>
            {currentStep === "details" && (
              <div className="space-y-4">
                <div>
                  <Label htmlFor="title">Product Title *</Label>
                  <Input
                    id="title"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g., The Complete Email Marketing Masterclass"
                  />
                </div>

                <div>
                  <Label htmlFor="niche">Niche *</Label>
                  <Input
                    id="niche"
                    value={niche}
                    onChange={(e) => setNiche(e.target.value)}
                    placeholder="e.g., Email Marketing, Fitness, Photography"
                  />
                </div>

                <div>
                  <Label htmlFor="targetAudience">Target Audience *</Label>
                  <Input
                    id="targetAudience"
                    value={targetAudience}
                    onChange={(e) => setTargetAudience(e.target.value)}
                    placeholder="e.g., Small business owners, Beginners"
                  />
                </div>

                <div>
                  <Label htmlFor="description">Description (Optional)</Label>
                  <Textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Additional context about your product..."
                    rows={4}
                  />
                </div>

                <Button
                  onClick={handleCreateProduct}
                  disabled={createProductMutation.isPending || generateOutlineMutation.isPending}
                  className="w-full"
                >
                  {createProductMutation.isPending || generateOutlineMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Course Outline...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Create Product & Generate Outline
                    </>
                  )}
                </Button>
              </div>
            )}

            {currentStep === "outline" && outline && (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-green-800 font-medium">
                    ✓ Course outline generated with {outline.modules.length} modules
                  </p>
                  <p className="text-green-600 text-sm mt-1">
                    Estimated total duration: {outline.totalEstimatedDuration} minutes
                  </p>
                </div>

                <div className="space-y-3">
                  {outline.modules.map((module: any, index: number) => (
                    <Card key={index}>
                      <CardHeader>
                        <CardTitle className="text-lg">
                          Module {index + 1}: {module.title}
                        </CardTitle>
                        <CardDescription>{module.description}</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm font-medium mb-2">Learning Objectives:</p>
                        <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                          {module.learningObjectives.map((obj: string, i: number) => (
                            <li key={i}>{obj}</li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                <Button
                  onClick={handleGenerateModules}
                  disabled={generateModulesMutation.isPending}
                  className="w-full"
                >
                  {generateModulesMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Module Content...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Generate All Module Content
                    </>
                  )}
                </Button>
              </div>
            )}

            {currentStep === "modules" && modulesGenerated && (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-green-800 font-medium">
                    ✓ All modules generated successfully!
                  </p>
                  <p className="text-green-600 text-sm mt-1">
                    Each module includes:
                  </p>
                  <ul className="list-disc list-inside text-green-600 text-sm mt-2 space-y-1">
                    <li>On-screen document (formatted for video display)</li>
                    <li>Narration script with hooks, cues, and anecdote prompts</li>
                    <li>Estimated duration</li>
                  </ul>
                </div>

                <Button
                  onClick={handleGenerateAssets}
                  disabled={generateAssetsMutation.isPending}
                  className="w-full"
                >
                  {generateAssetsMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Assets...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Generate Assets & Frameworks
                    </>
                  )}
                </Button>
              </div>
            )}

            {currentStep === "assets" && !assetsGenerated && (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-blue-800 font-medium">
                    Ready to generate assets and frameworks
                  </p>
                  <p className="text-blue-600 text-sm mt-1">
                    AI will create worksheets, templates, checklists, and guides to complement your course
                  </p>
                </div>

                <Button
                  onClick={handleGenerateAssets}
                  disabled={generateAssetsMutation.isPending}
                  className="w-full"
                >
                  {generateAssetsMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Assets...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Generate Assets & Frameworks
                    </>
                  )}
                </Button>
              </div>
            )}

            {currentStep === "assets" && assetsGenerated && (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-green-800 font-medium">
                    ✓ Assets and frameworks generated!
                  </p>
                  <p className="text-green-600 text-sm mt-1">
                    Your product now includes worksheets, templates, checklists, and guides
                  </p>
                </div>

                <Button onClick={() => setCurrentStep("pricing")} className="w-full">
                  Continue to Pricing
                </Button>
              </div>
            )}

            {currentStep === "pricing" && (
              <div className="space-y-4">
                <p className="text-sm text-gray-600">
                  Set pricing for your digital product. You can offer monthly, yearly, or lifetime access.
                </p>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="priceMonthly">Monthly ($)</Label>
                    <Input
                      id="priceMonthly"
                      type="number"
                      step="0.01"
                      value={priceMonthly}
                      onChange={(e) => setPriceMonthly(e.target.value)}
                      placeholder="29.99"
                    />
                  </div>

                  <div>
                    <Label htmlFor="priceYearly">Yearly ($)</Label>
                    <Input
                      id="priceYearly"
                      type="number"
                      step="0.01"
                      value={priceYearly}
                      onChange={(e) => setPriceYearly(e.target.value)}
                      placeholder="199.99"
                    />
                  </div>

                  <div>
                    <Label htmlFor="priceLifetime">Lifetime ($)</Label>
                    <Input
                      id="priceLifetime"
                      type="number"
                      step="0.01"
                      value={priceLifetime}
                      onChange={(e) => setPriceLifetime(e.target.value)}
                      placeholder="499.99"
                    />
                  </div>
                </div>

                <Button
                  onClick={handleSavePricing}
                  disabled={updateProductMutation.isPending}
                  className="w-full"
                >
                  {updateProductMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    "Save Pricing & Continue"
                  )}
                </Button>
              </div>
            )}

            {currentStep === "review" && (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-blue-800 font-medium">
                    🎉 Your digital product is ready!
                  </p>
                  <p className="text-blue-600 text-sm mt-1">
                    Review the details below and publish when ready.
                  </p>
                </div>

                <div className="space-y-2">
                  <div>
                    <p className="text-sm font-medium">Product Title</p>
                    <p className="text-gray-600">{title}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Niche</p>
                    <p className="text-gray-600">{niche}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Target Audience</p>
                    <p className="text-gray-600">{targetAudience}</p>
                  </div>
                  {outline && (
                    <div>
                      <p className="text-sm font-medium">Course Modules</p>
                      <p className="text-gray-600">{outline.modules.length} modules</p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm font-medium">Pricing</p>
                    <p className="text-gray-600">
                      {priceMonthly && `Monthly: $${priceMonthly} `}
                      {priceYearly && `Yearly: $${priceYearly} `}
                      {priceLifetime && `Lifetime: $${priceLifetime}`}
                    </p>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={() => navigate("/dashboard")}
                    variant="outline"
                    className="flex-1"
                  >
                    Save as Draft
                  </Button>
                  <Button
                    onClick={handlePublish}
                    disabled={updateProductMutation.isPending}
                    className="flex-1"
                  >
                    {updateProductMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Publishing...
                      </>
                    ) : (
                      "Publish Product"
                    )}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
