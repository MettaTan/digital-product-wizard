import { invokeLLM } from "./_core/llm";

/**
 * AI Content Generation Service
 * Generates course outlines, module content (on-screen docs + scripts), and assets
 */

// ===== TYPES =====

export interface CourseOutlineModule {
  title: string;
  description: string;
  learningObjectives: string[];
}

export interface CourseOutline {
  modules: CourseOutlineModule[];
  totalEstimatedDuration: number; // in minutes
}

export interface ModuleContent {
  onScreenDoc: string; // Markdown formatted document
  script: string; // Narration script with hooks and cues
  estimatedDuration: number; // in minutes
}

export interface AssetContent {
  title: string;
  type: "worksheet" | "template" | "checklist" | "framework" | "guide";
  content: string; // Markdown formatted content
}

// ===== COURSE OUTLINE GENERATION =====

export async function generateCourseOutline(params: {
  productTitle: string;
  niche: string;
  targetAudience: string;
  description?: string;
}): Promise<CourseOutline> {
  const { productTitle, niche, targetAudience, description } = params;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content: `You are an expert course designer who creates comprehensive, engaging digital course outlines. 
Your courses are practical, actionable, and designed to deliver real transformation for students.
You structure courses with clear progression, building from fundamentals to advanced concepts.`,
      },
      {
        role: "user",
        content: `Create a detailed course outline for a digital product with the following details:

Product Title: ${productTitle}
Niche: ${niche}
Target Audience: ${targetAudience}
${description ? `Additional Context: ${description}` : ""}

Create 6-10 modules that will take students from beginner to competent. Each module should:
- Have a clear, compelling title
- Include a detailed description of what will be covered
- List 3-5 specific learning objectives
- Build logically on previous modules

Make this course practical and transformation-focused, not just theoretical.`,
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "course_outline",
        strict: true,
        schema: {
          type: "object",
          properties: {
            modules: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  title: { type: "string" },
                  description: { type: "string" },
                  learningObjectives: {
                    type: "array",
                    items: { type: "string" },
                  },
                },
                required: ["title", "description", "learningObjectives"],
                additionalProperties: false,
              },
            },
            totalEstimatedDuration: { type: "integer" },
          },
          required: ["modules", "totalEstimatedDuration"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content;
  if (!content || typeof content !== 'string') throw new Error("No content generated");

  return JSON.parse(content);
}

// ===== MODULE CONTENT GENERATION =====

export async function generateModuleContent(params: {
  productTitle: string;
  niche: string;
  moduleTitle: string;
  moduleDescription: string;
  learningObjectives: string[];
  moduleNumber: number;
  totalModules: number;
}): Promise<ModuleContent> {
  const {
    productTitle,
    niche,
    moduleTitle,
    moduleDescription,
    learningObjectives,
    moduleNumber,
    totalModules,
  } = params;

  // Generate on-screen document
  const docResponse = await invokeLLM({
    messages: [
      {
        role: "system",
        content: `You are an expert content creator who writes engaging, visual-friendly course content.
Your content is designed to be displayed on-screen during video recordings (like a Google Doc).
You write in a clear, scannable format with headers, bullet points, examples, and actionable steps.
Your content is practical, not overly academic, and includes real-world examples.`,
      },
      {
        role: "user",
        content: `Create the on-screen document content for this course module:

Course: ${productTitle} (${niche})
Module ${moduleNumber} of ${totalModules}: ${moduleTitle}
Description: ${moduleDescription}
Learning Objectives:
${learningObjectives.map((obj) => `- ${obj}`).join("\n")}

Create a comprehensive, well-structured document that will be displayed on screen during video recording.
Include:
- Clear section headers
- Key concepts explained simply
- Practical examples
- Step-by-step instructions where relevant
- Important tips or warnings
- Summary/key takeaways

Format in Markdown. Make it visual and easy to follow on screen.
Length: 800-1200 words.`,
      },
    ],
  });

  const onScreenDoc = typeof docResponse.choices[0].message.content === 'string' 
    ? docResponse.choices[0].message.content 
    : "";

  // Generate narration script
  const scriptResponse = await invokeLLM({
    messages: [
      {
        role: "system",
        content: `You are an expert video script writer who creates engaging narration scripts for course videos.
Your scripts are conversational, engaging, and designed to accompany on-screen content.
You include hooks, transitions, cues for the instructor, and prompts for personal anecdotes.
Your scripts make the instructor sound natural, not robotic.`,
      },
      {
        role: "user",
        content: `Create a narration script for this course module video:

Module: ${moduleTitle}
Description: ${moduleDescription}

The instructor will be scrolling through an on-screen document while narrating.
Here's the on-screen content:

${onScreenDoc}

Create a complete narration script that:
- Opens with a hook to grab attention
- Guides the viewer through the on-screen content naturally
- Includes [CUE: ...] markers for instructor actions (scroll, pause, emphasize)
- Includes [ANECDOTE PROMPT: ...] suggestions for personal stories
- Uses conversational, engaging language
- Includes smooth transitions between sections
- Ends with a clear call-to-action or transition to next module

Make it feel natural and engaging, not like reading a textbook.`,
      },
    ],
  });

  const script = typeof scriptResponse.choices[0].message.content === 'string'
    ? scriptResponse.choices[0].message.content
    : "";

  // Estimate duration based on content length
  const wordCount = onScreenDoc.split(/\s+/).length;
  const estimatedDuration = Math.ceil(wordCount / 150); // ~150 words per minute

  return {
    onScreenDoc,
    script,
    estimatedDuration,
  };
}

// ===== ASSET GENERATION =====

export async function generateAssets(params: {
  productTitle: string;
  niche: string;
  targetAudience: string;
  courseOutline: CourseOutlineModule[];
}): Promise<AssetContent[]> {
  const { productTitle, niche, targetAudience, courseOutline } = params;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content: `You are an expert at creating practical, high-value course assets and frameworks.
You create worksheets, templates, checklists, and frameworks that students can actually use.
Your assets are actionable, not just informational - they help students implement what they learn.`,
      },
      {
        role: "user",
        content: `Create 3-5 valuable assets/frameworks for this digital product:

Product: ${productTitle}
Niche: ${niche}
Target Audience: ${targetAudience}

Course Modules:
${courseOutline.map((m, i) => `${i + 1}. ${m.title}`).join("\n")}

Create a variety of assets such as:
- Worksheets (fill-in-the-blank exercises)
- Templates (ready-to-use frameworks)
- Checklists (step-by-step action lists)
- Frameworks (strategic planning tools)
- Guides (quick reference materials)

Each asset should be practical and directly support the course content.
Format each in Markdown with clear structure.`,
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "course_assets",
        strict: true,
        schema: {
          type: "object",
          properties: {
            assets: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  title: { type: "string" },
                  type: {
                    type: "string",
                    enum: ["worksheet", "template", "checklist", "framework", "guide"],
                  },
                  content: { type: "string" },
                },
                required: ["title", "type", "content"],
                additionalProperties: false,
              },
            },
          },
          required: ["assets"],
          additionalProperties: false,
        },
      },
    },
  });

  const content = response.choices[0].message.content;
  if (!content || typeof content !== 'string') throw new Error("No content generated");

  const parsed = JSON.parse(content);
  return parsed.assets;
}

// ===== COMMUNITY SETUP HELPER =====

export async function generateCommunitySetup(params: {
  productTitle: string;
  niche: string;
  platform: string; // discord, circle, slack, etc.
}): Promise<string> {
  const { productTitle, niche, platform } = params;

  const response = await invokeLLM({
    messages: [
      {
        role: "system",
        content: `You are an expert at setting up and managing online communities for digital products.
You provide clear, actionable instructions for community setup and engagement.`,
      },
      {
        role: "user",
        content: `Create setup instructions for a ${platform} community for this digital product:

Product: ${productTitle}
Niche: ${niche}
Platform: ${platform}

Provide:
1. Recommended channel/category structure
2. Welcome message template
3. Community rules/guidelines
4. Engagement strategies
5. Moderation tips

Format in Markdown.`,
      },
    ],
  });

  return typeof response.choices[0].message.content === 'string'
    ? response.choices[0].message.content
    : "";
}
