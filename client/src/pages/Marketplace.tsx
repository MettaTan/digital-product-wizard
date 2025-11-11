import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, ShoppingCart, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/_core/hooks/useAuth";
import { getLoginUrl } from "@/const";

export default function Marketplace() {
  const { isAuthenticated, user } = useAuth();

  // For now, we'll show all published products
  // In a real app, you might want a separate public products endpoint
  const { data: products, isLoading } = trpc.products.list.useQuery();
  const { data: myPurchases } = trpc.products.myPurchases.useQuery(undefined, {
    enabled: isAuthenticated,
  });

  const createCheckoutMutation = trpc.products.createCheckout.useMutation({
    onSuccess: (data) => {
      window.open(data.checkoutUrl, "_blank");
      toast.info("Redirecting to checkout...");
    },
    onError: (error) => {
      toast.error(error.message || "Failed to create checkout");
    },
  });

  const handlePurchase = (productId: number, tier: "monthly" | "yearly" | "lifetime") => {
    if (!isAuthenticated) {
      toast.error("Please sign in to purchase");
      window.location.href = getLoginUrl();
      return;
    }

    createCheckoutMutation.mutate({ productId, tier });
  };

  const hasPurchased = (productId: number) => {
    return myPurchases?.some((p) => p.productId === productId && p.status === "active");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-purple-600" />
      </div>
    );
  }

  const publishedProducts = products?.filter((p) => p.status === "published") || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 py-12">
      <div className="container">
        <div className="mb-8">
          <h1 className="text-4xl font-bold">Marketplace</h1>
          <p className="text-gray-600 mt-2">Browse and purchase digital products</p>
        </div>

        {publishedProducts.length === 0 ? (
          <Card className="text-center py-12">
            <CardContent>
              <ShoppingCart className="h-16 w-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-xl font-semibold mb-2">No products available yet</h3>
              <p className="text-gray-600">Check back soon for new digital products!</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {publishedProducts.map((product) => {
              const purchased = hasPurchased(product.id);

              return (
                <Card key={product.id} className="flex flex-col">
                  <CardHeader>
                    <CardTitle>{product.title}</CardTitle>
                    <CardDescription className="line-clamp-2">
                      {product.description || product.niche}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col">
                    <div className="space-y-2 text-sm text-gray-600 mb-4">
                      <div>
                        <Badge variant="outline">{product.niche}</Badge>
                      </div>
                      <div>
                        <span className="font-medium">Target:</span> {product.targetAudience}
                      </div>
                    </div>

                    {purchased ? (
                      <div className="mt-auto">
                        <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center gap-2">
                          <CheckCircle2 className="h-5 w-5 text-green-600" />
                          <span className="text-green-800 font-medium">Already Purchased</span>
                        </div>
                        <Button asChild className="w-full mt-2">
                          <a href={`/my-products/${product.id}`}>Access Product</a>
                        </Button>
                      </div>
                    ) : (
                      <div className="mt-auto space-y-2">
                        {product.priceMonthly && (
                          <Button
                            variant="outline"
                            className="w-full"
                            onClick={() => handlePurchase(product.id, "monthly")}
                            disabled={createCheckoutMutation.isPending}
                          >
                            Monthly - ${(product.priceMonthly / 100).toFixed(2)}
                          </Button>
                        )}
                        {product.priceYearly && (
                          <Button
                            variant="outline"
                            className="w-full"
                            onClick={() => handlePurchase(product.id, "yearly")}
                            disabled={createCheckoutMutation.isPending}
                          >
                            Yearly - ${(product.priceYearly / 100).toFixed(2)}
                          </Button>
                        )}
                        {product.priceLifetime && (
                          <Button
                            className="w-full"
                            onClick={() => handlePurchase(product.id, "lifetime")}
                            disabled={createCheckoutMutation.isPending}
                          >
                            <ShoppingCart className="mr-2 h-4 w-4" />
                            Lifetime - ${(product.priceLifetime / 100).toFixed(2)}
                          </Button>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
