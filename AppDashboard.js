import { StatusBar } from 'expo-status-bar';
import { MaterialIcons } from '@expo/vector-icons';
import { useState } from 'react';
import {
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar as NativeStatusBar,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import CameraDetectionScreen from './CameraDetectionScreen';

const COLORS = {
  primary: '#256AF4',
  bgDark: '#101622',
  surfaceDark: '#0F172ACC',
  headerBorder: '#1F2937',
  textStrong: '#F1F5F9',
  textBody: '#CBD5E1',
  textMuted: '#94A3B8',
  textSubtle: '#64748B',
  success: '#10B981',
};

export default function AppDashboard() {
  const { width, height } = useWindowDimensions();
  const [screen, setScreen] = useState('dashboard');
  const isCompact = width < 360 || height < 700;
  const horizontalPadding = Math.min(24, Math.max(14, width * 0.055));
  const heroSize = width < 360 ? 208 : width < 420 ? 236 : 256;
  const safeTopInset = Platform.OS === 'android' ? NativeStatusBar.currentHeight ?? 0 : 0;

  const styles = createStyles({
    heroSize,
    horizontalPadding,
    isCompact,
    safeTopInset,
  });

  if (screen === 'camera') {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <CameraDetectionScreen onBack={() => setScreen('dashboard')} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />

      <View style={styles.header}>
        <View style={styles.brandRow}>
          <View style={styles.logoBox}>
            <MaterialIcons name="back-hand" size={20} color="#FFFFFF" />
          </View>
          <Text style={styles.brandText}>TinigKamay</Text>
        </View>
        <Pressable
          accessibilityLabel="Notifications"
          android_ripple={{ color: '#FFFFFF22' }}
          onPress={() => {}}
          style={styles.iconButton}
        >
          <MaterialIcons name="notifications-none" size={22} color={COLORS.textMuted} />
          <View style={styles.notificationDotOuter}>
            <View style={styles.notificationDotInner} />
          </View>
        </Pressable>
      </View>

      <ScrollView
        style={styles.main}
        contentContainerStyle={styles.mainContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.container}>
          <View style={styles.heroSection}>
            <View style={styles.heroGlow} />
            <Pressable
              accessibilityHint="Opens camera recognition screen."
              accessibilityLabel="Start Recognition"
              android_ripple={{ color: '#FFFFFF22' }}
              onPress={() => setScreen('camera')}
              style={({ pressed }) => [
                styles.heroButton,
                pressed ? styles.heroButtonPressed : null,
              ]}
            >
              <MaterialIcons name="record-voice-over" size={58} color="#FFFFFF" />
              <Text style={styles.heroTitle}>Start Recognition</Text>
            </Pressable>
            <Text style={styles.heroCaption}>
              Position your hands in front of the camera to begin translating sign language to
              speech.
            </Text>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>System Status</Text>
            <View style={styles.statusList}>
              <StatusRow icon="videocam" label="Camera" status="Ready" tone="success" />
              <StatusRow icon="psychology" label="Model" status="YOLOv11" tone="tag" />
              <StatusRow icon="volume-up" label="Output Mode" status="Text & Speech" tone="text" />
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function StatusRow({ icon, label, status, tone }) {
  return (
    <View style={stylesShared.statusCard}>
      <View style={stylesShared.statusLeft}>
        <View style={stylesShared.iconTile}>
          <MaterialIcons name={icon} size={20} color={COLORS.primary} />
        </View>
        <Text style={stylesShared.statusName}>{label}</Text>
      </View>
      {tone === 'success' ? (
        <View style={stylesShared.statusRight}>
          <View style={stylesShared.readyDot} />
          <Text style={stylesShared.statusReady}>{status}</Text>
        </View>
      ) : tone === 'tag' ? (
        <Text style={stylesShared.statusTag}>{status}</Text>
      ) : (
        <Text style={stylesShared.statusNeutral}>{status}</Text>
      )}
    </View>
  );
}

const createStyles = ({ heroSize, horizontalPadding, isCompact, safeTopInset }) =>
  StyleSheet.create({
    root: { flex: 1, backgroundColor: COLORS.bgDark, paddingTop: safeTopInset },
    header: {
      paddingHorizontal: horizontalPadding,
      paddingTop: isCompact ? 10 : 14,
      paddingBottom: 12,
      borderBottomWidth: 1,
      borderBottomColor: COLORS.headerBorder,
      backgroundColor: COLORS.bgDark,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    brandRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    logoBox: {
      width: 38,
      height: 38,
      borderRadius: 10,
      backgroundColor: COLORS.primary,
      alignItems: 'center',
      justifyContent: 'center',
    },
    brandText: { color: COLORS.textStrong, fontSize: isCompact ? 21 : 23, fontWeight: '700' },
    iconButton: {
      width: 38,
      height: 38,
      borderRadius: 19,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#111827',
      position: 'relative',
    },
    notificationDotOuter: {
      position: 'absolute',
      top: 8,
      right: 9,
      width: 8,
      height: 8,
      borderRadius: 4,
      backgroundColor: '#256AF4AA',
      alignItems: 'center',
      justifyContent: 'center',
    },
    notificationDotInner: {
      width: 5,
      height: 5,
      borderRadius: 3,
      backgroundColor: COLORS.primary,
    },
    main: { flex: 1 },
    mainContent: { flexGrow: 1 },
    container: {
      width: '100%',
      maxWidth: 500,
      alignSelf: 'center',
      paddingHorizontal: horizontalPadding,
      paddingTop: isCompact ? 18 : 24,
      paddingBottom: 20,
      flex: 1,
    },
    heroSection: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: isCompact ? 30 : 42,
      position: 'relative',
    },
    heroGlow: {
      position: 'absolute',
      width: heroSize + 30,
      height: heroSize + 30,
      borderRadius: (heroSize + 30) / 2,
      backgroundColor: '#256AF433',
    },
    heroButton: {
      width: heroSize,
      height: heroSize,
      borderRadius: heroSize / 2,
      backgroundColor: COLORS.primary,
      alignItems: 'center',
      justifyContent: 'center',
      paddingHorizontal: 24,
    },
    heroButtonPressed: { transform: [{ scale: 0.98 }], opacity: 0.95 },
    heroTitle: {
      color: '#FFFFFF',
      fontSize: isCompact ? 18 : 20,
      fontWeight: '700',
      textTransform: 'uppercase',
      textAlign: 'center',
      letterSpacing: 0.8,
      marginTop: 12,
    },
    heroCaption: {
      marginTop: 22,
      color: COLORS.textMuted,
      fontSize: 13,
      fontWeight: '500',
      lineHeight: 19,
      textAlign: 'center',
      maxWidth: 360,
    },
    section: { marginBottom: 18 },
    sectionTitle: {
      color: COLORS.textSubtle,
      fontSize: 12,
      fontWeight: '600',
      letterSpacing: 1.2,
      textTransform: 'uppercase',
      marginBottom: 10,
    },
    statusList: { gap: 10 },
  });

const stylesShared = StyleSheet.create({
  statusCard: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.headerBorder,
    backgroundColor: COLORS.surfaceDark,
    paddingVertical: 14,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statusLeft: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  iconTile: {
    width: 40,
    height: 40,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#256AF433',
  },
  statusName: { color: COLORS.textStrong, fontSize: 15, fontWeight: '500' },
  statusRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  readyDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.success,
  },
  statusReady: { color: COLORS.success, fontSize: 14, fontWeight: '700' },
  statusNeutral: { color: COLORS.textBody, fontSize: 14, fontWeight: '700' },
  statusTag: {
    color: COLORS.textBody,
    backgroundColor: '#1E293B',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    fontSize: 12,
    fontWeight: '600',
  },
});
