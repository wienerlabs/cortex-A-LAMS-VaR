/**
 * Notification Service
 *
 * Sends notifications to PM and team when:
 * - Retraining is triggered
 * - Retraining completed
 * - New model deployed
 * - Retraining failed
 *
 * Supports multiple notification channels: email, Slack, logs.
 */

import { logger } from '../logger.js';

// ============= TYPES =============

export type NotificationType = 
  | 'RETRAINING_STARTED'
  | 'RETRAINING_COMPLETED'
  | 'NEW_MODEL_DEPLOYED'
  | 'RETRAINING_FAILED'
  | 'PERFORMANCE_DEGRADATION'
  | 'MODEL_DRIFT_DETECTED';

export type NotificationChannel = 'email' | 'slack' | 'log';

export interface Notification {
  type: NotificationType;
  modelName: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp?: Date;
}

export interface NotificationConfig {
  enabled: boolean;
  channels: NotificationChannel[];
  pmEmail?: string;
  slackWebhookUrl?: string;
  notifyOn: NotificationType[];
}

export interface NotificationResult {
  success: boolean;
  channel: NotificationChannel;
  error?: string;
}

// ============= CONSTANTS =============

const DEFAULT_CONFIG: NotificationConfig = {
  enabled: true,
  channels: ['log'],
  notifyOn: [
    'RETRAINING_STARTED',
    'RETRAINING_COMPLETED',
    'NEW_MODEL_DEPLOYED',
    'RETRAINING_FAILED',
    'PERFORMANCE_DEGRADATION',
    'MODEL_DRIFT_DETECTED',
  ],
};

// ============= NOTIFICATION SERVICE CLASS =============

class NotificationService {
  private config: NotificationConfig;

  constructor(config: Partial<NotificationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    logger.info('[NotificationService] Initialized', { 
      channels: this.config.channels,
      notifyOn: this.config.notifyOn,
    });
  }

  /**
   * Send a notification through all configured channels
   */
  async notify(notification: Notification): Promise<NotificationResult[]> {
    if (!this.config.enabled) {
      logger.debug('[NotificationService] Disabled, skipping notification');
      return [];
    }

    // Check if we should notify for this type
    if (!this.config.notifyOn.includes(notification.type)) {
      logger.debug('[NotificationService] Notification type not in notifyOn list', {
        type: notification.type,
      });
      return [];
    }

    const results: NotificationResult[] = [];
    const enrichedNotification = {
      ...notification,
      timestamp: notification.timestamp || new Date(),
    };

    for (const channel of this.config.channels) {
      try {
        await this.sendToChannel(channel, enrichedNotification);
        results.push({ success: true, channel });
      } catch (error) {
        results.push({ 
          success: false, 
          channel, 
          error: String(error),
        });
      }
    }

    return results;
  }

  /**
   * Send notification to a specific channel
   */
  private async sendToChannel(
    channel: NotificationChannel, 
    notification: Notification
  ): Promise<void> {
    switch (channel) {
      case 'log':
        this.sendToLog(notification);
        break;
      case 'email':
        await this.sendToEmail(notification);
        break;
      case 'slack':
        await this.sendToSlack(notification);
        break;
    }
  }

  /**
   * Send notification to log (always works)
   */
  private sendToLog(notification: Notification): void {
    const logLevel = this.getLogLevel(notification.type);
    const logData = {
      type: notification.type,
      modelName: notification.modelName,
      message: notification.message,
      details: notification.details,
    };

    switch (logLevel) {
      case 'error':
        logger.error('[NotificationService]', logData);
        break;
      case 'warn':
        logger.warn('[NotificationService]', logData);
        break;
      default:
        logger.info('[NotificationService]', logData);
    }
  }

  /**
   * Get appropriate log level for notification type
   */
  private getLogLevel(type: NotificationType): 'info' | 'warn' | 'error' {
    switch (type) {
      case 'RETRAINING_FAILED':
        return 'error';
      case 'PERFORMANCE_DEGRADATION':
      case 'MODEL_DRIFT_DETECTED':
        return 'warn';
      default:
        return 'info';
    }
  }

  /**
   * Send notification via email
   */
  private async sendToEmail(notification: Notification): Promise<void> {
    if (!this.config.pmEmail) {
      logger.warn('[NotificationService] Email channel configured but no pmEmail set');
      return;
    }

    // Format email content
    const subject = `[Cortex ML] ${notification.type}: ${notification.modelName}`;
    const body = this.formatEmailBody(notification);

    logger.info('[NotificationService] Would send email', {
      to: this.config.pmEmail,
      subject,
      // In production, integrate with email service (SendGrid, SES, etc.)
    });

    // TODO: Integrate with email service
    // await emailService.send({
    //   to: this.config.pmEmail,
    //   subject,
    //   body,
    // });
  }

  /**
   * Format email body
   */
  private formatEmailBody(notification: Notification): string {
    const lines = [
      `Model: ${notification.modelName}`,
      `Event: ${notification.type}`,
      `Time: ${notification.timestamp?.toISOString()}`,
      '',
      `Message: ${notification.message}`,
    ];

    if (notification.details) {
      lines.push('', 'Details:', JSON.stringify(notification.details, null, 2));
    }

    return lines.join('\n');
  }

  /**
   * Send notification via Slack webhook
   */
  private async sendToSlack(notification: Notification): Promise<void> {
    if (!this.config.slackWebhookUrl) {
      logger.warn('[NotificationService] Slack channel configured but no webhook URL set');
      return;
    }

    const payload = this.formatSlackPayload(notification);

    try {
      const response = await fetch(this.config.slackWebhookUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Slack webhook returned ${response.status}`);
      }

      logger.info('[NotificationService] Slack notification sent', {
        modelName: notification.modelName,
        type: notification.type,
      });
    } catch (error) {
      logger.error('[NotificationService] Slack notification failed', {
        error: String(error),
      });
      throw error;
    }
  }

  /**
   * Format Slack message payload
   */
  private formatSlackPayload(notification: Notification): Record<string, unknown> {
    const emoji = this.getEmoji(notification.type);
    const color = this.getColor(notification.type);

    return {
      attachments: [
        {
          color,
          blocks: [
            {
              type: 'header',
              text: {
                type: 'plain_text',
                text: `${emoji} ${notification.type}`,
              },
            },
            {
              type: 'section',
              fields: [
                {
                  type: 'mrkdwn',
                  text: `*Model:*\n${notification.modelName}`,
                },
                {
                  type: 'mrkdwn',
                  text: `*Time:*\n${notification.timestamp?.toISOString() || 'N/A'}`,
                },
              ],
            },
            {
              type: 'section',
              text: {
                type: 'mrkdwn',
                text: notification.message,
              },
            },
          ],
        },
      ],
    };
  }

  /**
   * Get emoji for notification type
   */
  private getEmoji(type: NotificationType): string {
    switch (type) {
      case 'RETRAINING_STARTED':
        return '🔄';
      case 'RETRAINING_COMPLETED':
        return '✅';
      case 'NEW_MODEL_DEPLOYED':
        return '🚀';
      case 'RETRAINING_FAILED':
        return '❌';
      case 'PERFORMANCE_DEGRADATION':
        return '📉';
      case 'MODEL_DRIFT_DETECTED':
        return '⚠️';
      default:
        return '📣';
    }
  }

  /**
   * Get color for notification type
   */
  private getColor(type: NotificationType): string {
    switch (type) {
      case 'RETRAINING_FAILED':
        return '#dc3545';  // Red
      case 'PERFORMANCE_DEGRADATION':
      case 'MODEL_DRIFT_DETECTED':
        return '#ffc107';  // Yellow
      case 'NEW_MODEL_DEPLOYED':
        return '#28a745';  // Green
      default:
        return '#17a2b8';  // Blue
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<NotificationConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('[NotificationService] Config updated', {
      channels: this.config.channels,
    });
  }
}

// ============= SINGLETON =============

let notificationServiceInstance: NotificationService | null = null;

export function getNotificationService(config?: Partial<NotificationConfig>): NotificationService {
  if (!notificationServiceInstance) {
    notificationServiceInstance = new NotificationService(config);
  }
  return notificationServiceInstance;
}

export const notificationService = getNotificationService();

export { NotificationService };

