// Dynamic Settings Pipeline for Open WebUI
// This pipeline automatically detects content type and applies optimal settings

module.exports = {
  name: "Dynamic Settings Pipeline",
  description: "Automatically detects content type and applies optimal LLM settings",
  pipeline: [
    {
      name: "Content Type Detection",
      description: "Analyzes the user's request to determine content type",
      execute: async (data, { logger }) => {
        logger.info("Detecting content type...");
        
        const userInput = data.input.text;
        
        // Keywords and patterns for each content type
        const contentTypes = [
          {
            name: "Creative Writing",
            keywords: ["story", "novel", "fiction", "creative", "narrative", "poem", "poetry", "tale", "write a story", "fantasy"],
            patterns: [
              /write a (story|poem|novel|fiction|tale)/i,
              /create a (story|poem|narrative|fiction)/i,
              /imagine a (story|scenario|world|character)/i
            ],
            settings: {
              temperature: 0.8,
              top_k: 80,
              top_p: 0.92,
              frequency_penalty: 0.3,
              presence_penalty: 0.3,
              max_tokens: 2000,
              repetition_penalty: 1.08
            }
          },
          {
            name: "Technical Writing",
            keywords: ["documentation", "report", "technical", "explain", "analysis", "research", "whitepaper", "specification"],
            patterns: [
              /write (documentation|a report|a technical|an analysis)/i,
              /explain (how|what|why)/i,
              /technical (details|specifications|requirements)/i
            ],
            settings: {
              temperature: 0.2,
              top_k: 30,
              top_p: 0.6,
              frequency_penalty: 0.2,
              presence_penalty: 0.2,
              max_tokens: 1500,
              repetition_penalty: 1.02
            }
          },
          {
            name: "Code Generation",
            keywords: ["code", "function", "programming", "algorithm", "script", "develop", "software", "class", "method"],
            patterns: [
              /write (a|some) code/i,
              /create (a function|a method|an algorithm|a class)/i,
              /implement (a|an)/i,
              /how to code/i,
              /\b(javascript|python|java|c\+\+|html|css|sql|php|ruby)\b/i
            ],
            settings: {
              temperature: 0.1,
              top_k: 20,
              top_p: 0.3,
              frequency_penalty: 0.05,
              presence_penalty: 0.05,
              max_tokens: 1500,
              repetition_penalty: 1.03
            }
          },
          {
            name: "Q&A/Factual",
            keywords: ["what is", "how does", "explain", "define", "who is", "when did", "why is", "where is", "fact", "information"],
            patterns: [
              /what (is|are|was|were)/i,
              /how (does|do|did)/i,
              /why (is|are|did)/i,
              /when (is|was|did)/i,
              /where (is|are|did)/i,
              /who (is|are|was|were)/i,
              /can you (explain|tell me about)/i
            ],
            settings: {
              temperature: 0.1,
              top_k: 10,
              top_p: 0.4,
              frequency_penalty: 0.1,
              presence_penalty: 0.1,
              max_tokens: 800,
              repetition_penalty: 1.01
            }
          },
          {
            name: "Business Communication",
            keywords: ["email", "business", "professional", "letter", "proposal", "memo", "report", "meeting", "corporate"],
            patterns: [
              /write (an email|a letter|a proposal|a memo|a report)/i,
              /draft (an email|a letter|a proposal|a memo|a report)/i,
              /business (proposal|plan|strategy|communication)/i,
              /professional (email|letter|communication)/i
            ],
            settings: {
              temperature: 0.3,
              top_k: 40,
              top_p: 0.7,
              frequency_penalty: 0.2,
              presence_penalty: 0.2,
              max_tokens: 1000,
              repetition_penalty: 1.04
            }
          },
          {
            name: "Summarization",
            keywords: ["summarize", "summary", "condense", "brief", "overview", "recap", "tldr", "in short"],
            patterns: [
              /summarize/i,
              /give (me|a) summary/i,
              /provide a (brief|summary|recap)/i,
              /tldr/i,
              /in (brief|short)/i,
              /can you condense/i
            ],
            settings: {
              temperature: 0.3,
              top_k: 30,
              top_p: 0.6,
              frequency_penalty: 0.4,
              presence_penalty: 0.3,
              max_tokens: 600,
              repetition_penalty: 1.08
            }
          },
          {
            name: "Brainstorming",
            keywords: ["ideas", "brainstorm", "suggestions", "alternatives", "options", "possibilities", "creative", "generate"],
            patterns: [
              /brainstorm/i,
              /generate (ideas|options|possibilities|alternatives)/i,
              /give me (ideas|suggestions|options)/i,
              /what are some (ideas|ways|approaches|methods)/i,
              /help me (think of|come up with)/i
            ],
            settings: {
              temperature: 0.9,
              top_k: 90,
              top_p: 0.95,
              frequency_penalty: 0.6,
              presence_penalty: 0.6,
              max_tokens: 1500,
              repetition_penalty: 1.02
            }
          },
          {
            name: "Instructional",
            keywords: ["how to", "guide", "tutorial", "steps", "instructions", "teach", "learn", "process", "explain how"],
            patterns: [
              /how to/i,
              /steps to/i,
              /guide (me|on|for)/i,
              /tutorial (on|for)/i,
              /instructions (for|on)/i,
              /teach me/i,
              /explain how to/i
            ],
            settings: {
              temperature: 0.4,
              top_k: 40,
              top_p: 0.7,
              frequency_penalty: 0.2,
              presence_penalty: 0.2,
              max_tokens: 1500,
              repetition_penalty: 1.03
            }
          }
        ];
        
        // Detect content type based on keywords and patterns
        let matchedType = null;
        let highestScore = 0;
        
        for (const type of contentTypes) {
          let score = 0;
          
          // Check keywords
          for (const keyword of type.keywords) {
            if (userInput.toLowerCase().includes(keyword.toLowerCase())) {
              score += 1;
            }
          }
          
          // Check patterns
          for (const pattern of type.patterns) {
            if (pattern.test(userInput)) {
              score += 2;
            }
          }
          
          if (score > highestScore) {
            highestScore = score;
            matchedType = type;
          }
        }
        
        // Default to Q&A if no clear match or low confidence
        if (!matchedType || highestScore < 2) {
          matchedType = contentTypes.find(type => type.name === "Q&A/Factual");
          logger.info("No clear content type detected, defaulting to Q&A/Factual");
        } else {
          logger.info(`Detected content type: ${matchedType.name} with confidence score: ${highestScore}`);
        }
        
        // Add content type and settings to data
        data.contentType = matchedType.name;
        data.optimizedSettings = matchedType.settings;
        
        return data;
      }
    },
    {
      name: "Apply Optimized Settings",
      description: "Applies the optimal settings based on detected content type",
      execute: async (data, { logger }) => {
        logger.info(`Applying optimized settings for ${data.contentType}`);
        
        // Apply settings to the LLM request
        data.llmConfig = {
          ...data.llmConfig, // Preserve any existing config
          ...data.optimizedSettings // Apply our optimized settings
        };
        
        // Add notification about content type to system prompt
        const contentTypeNotice = `[Note: This request has been identified as "${data.contentType}" content. Optimized settings have been applied to generate the best response for this type of content.]`;
        
        // Append to system message if it exists, or create one
        if (data.messages && Array.isArray(data.messages)) {
          // Find system message if it exists
          const systemMsgIndex = data.messages.findIndex(msg => msg.role === "system");
          
          if (systemMsgIndex >= 0) {
            // Append to existing system message
            data.messages[systemMsgIndex].content += "\n\n" + contentTypeNotice;
          } else {
            // Add new system message at the beginning
            data.messages.unshift({
              role: "system",
              content: contentTypeNotice
            });
          }
        }
        
        // Also prepare to add the notice to the final output
        data.contentTypeNotice = contentTypeNotice;
        
        return data;
      }
    },
    {
      name: "Process Response",
      description: "Processes the response to include information about content type",
      execute: async (data, { logger }) => {
        if (!data.response || !data.response.content) {
          logger.error("No response content found in data");
          return data;
        }
        
        // Add notice about content type and settings at the bottom of the response
        const settingsInfo = `
        
_Content Type: ${data.contentType}
This response was optimized using custom parameters for ${data.contentType.toLowerCase()} content._`;
        
        // Append the notice to the response
        data.response.content += settingsInfo;
        
        logger.info("Added content type information to response");
        return data;
      }
    }
  ]
};
